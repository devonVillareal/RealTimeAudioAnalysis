import pygame
import pyaudio as pa
import numpy as np
import queue
import math

#Clean up current code and make unified global variables, then implement controls to change them realtime as the app is running


# GLOBAL VARIABLES 
processed_audio = queue.Queue(maxsize=0) 
numParticls = 1000
Steps = (np.pi*2)/numParticls 
selector = 2
num_peaks = 2
X_var = 10 #Controls contorsion of the shapes, used to set initial value and then tied to bass
Y_var = 10
X_var_1 = 3
Y_var_1 = 3
bass_dampener = 0.1 #Controls how much bass is dampered out of the callback
treble_dampener = 1 #Controls how much treble is dampered out of the callback
bass_max,treble_max,mid_max = 1.0,1.0,1.0 #used in callback




#Things that tweak visuals:
rmsmultiplier = 100  #Scale rms
b_t_multiplier = 2  # Used to scale bass
treble_multiplier = 2 #Scale treble 
flux_multiplier = 1 #Scale flux 




fft_max = 1e-6 #Used to normalize fft within callback
DECAY = 0.95

# Audio Setup
CHUNK = 2048 * 2 
FORMAT = pa.paFloat32
CHANNELS = 1
RATE = 48000 
WIDTH, HEIGHT = 1000, 1000
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
DEFAULT_NOTE = None #When no note is being played 
MIN_PEAK_NORM = 0.05  # Minimum normalized amplitude that will be thought of as a peak 
THETAS = np.arange(0, np.pi*2, Steps) 
WELL_SMOOTH = 0.2

# Memory
prev_bassfft, prevtreblefft, prevmidfft = None, None, None
prev_rms, prev_bass, prev_treble, prev_peaks = 0, 10, 10, {}
prevbass_flux, prevtreble_flux, prevmid_flux = 0.0,0.0,0.0 #Measures when things such as bass or treble pop

#Smoothing
bg_fade_value = 1.0


class Dot: 
    def __init__(self, x, y, index): 
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.size = 2.2
        self.pulse_offset = index * 0.1 

    def changeSpeed(self, gravity_x, gravity_y, screen, rms, color):
        x_diff = (gravity_x - self.x)
        y_diff = (gravity_y - self.y)
        scale_factor_move = 6.324 / (np.sqrt(2) * WIDTH)
        strength = (np.sqrt((x_diff**2) + (y_diff**2))) 
        if strength > 1:
            self.vx = (x_diff / strength) * ((strength * scale_factor_move)**1.27) + x_diff / 4
            self.vy = (y_diff / strength) * ((strength * scale_factor_move)**1.27) + y_diff / 4
            self.x += self.vx
            self.y += self.vy
        else:
            self.x, self.y = gravity_x, gravity_y
        self.draw(screen, rms, color)

    def draw(self, screen, rms, color): 
        pulse = math.sin((pygame.time.get_ticks() * 0.005) + self.pulse_offset)
        
        # --- CIRCULAR DIFFUSION GLOW ---
        glow_radius = int(self.size * (5 + pulse * 2) + (rms * 30))
        if glow_radius < 5: glow_radius = 5
        
        surf_size = glow_radius * 1
        glow_surface = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
        
        glow_color = (*color, 30) 
        pygame.draw.circle(glow_surface, glow_color, (glow_radius, glow_radius), glow_radius)

        screen.blit(glow_surface, (int(self.x) - glow_radius, int(self.y) - glow_radius))
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), int(self.size), width=0)

class Gravity_Well: 
    def __init__(self, x, y, Dot): 
        self.center_x = x
        self.center_y = y
        self.Particle = Dot
        
    def update(self, new_x, new_y, screen, rms, color): 
        self.center_x = new_x
        self.center_y = new_y
        self.Particle.changeSpeed(self.center_x, self.center_y, screen, rms, color)

# --- PYGAME SETUP ---
pygame.init()
pygame.display.set_caption("Music Visualizer") 
screen = pygame.display.set_mode((WIDTH, HEIGHT),pygame.RESIZABLE) 

def callback(in_data, frame_count, time_info, status):
    global fft_max, prev_bassfft, prevtreblefft, prevmidfft,bass_max,treble_max,mid_max,prevbass_flux,prevmid_flux,prevtreble_flux
    

    #Get audio data
    audio_data = np.frombuffer(in_data, dtype=np.float32)

    #Get fft and frequencies and loudness
    fft = np.abs(np.fft.rfft(audio_data))
    freqs = np.fft.rfftfreq(len(audio_data), 1 / RATE)
    rms = np.sqrt(np.mean(audio_data**2))


    bass_mask = (freqs >= 20) & (freqs <= 250) #Only bass frequencies
    treble_mask = (freqs >= 4000) & (freqs <= 10000) #only Treble frequencies
    mid_mask = (freqs >= 250) & (freqs <= 4000) #only mid Frequencies
    
    freq_mask = (freqs >= 20) & (freqs <= 10000) #Make sure frequency array contains only frequencies counted

    #Apply bounds
    bassfft_raw = fft[bass_mask]
    bassfreq_bound = freqs[bass_mask]

    treblefft_raw = fft[treble_mask]
    treblefreq_bound = freqs[treble_mask]

    midfft_raw= fft[mid_mask]
    midfreq_bound = freqs[mid_mask]

    freqs_bound = freqs[freq_mask]
   

    # Captures current loudest frame spike and decays slowly during silence
    raw_b_max = np.max(bassfft_raw) if len(bassfft_raw) > 0 else 0.01
    raw_m_max = np.max(midfft_raw) if len(midfft_raw) > 0 else 0.01
    raw_t_max = np.max(treblefft_raw) if len(treblefft_raw) > 0 else 0.01

    bass_max   = max(raw_b_max,   max(bass_max * DECAY, 0.01))
    mid_max    = max(raw_m_max,    max(mid_max * DECAY, 0.01))
    treble_max = max(raw_t_max, max(treble_max * DECAY, 0.01))



    # Normalize arrays to a perfect 0.0 to 1.0 baseline
    bassfft_boud   = bassfft_raw   / (bass_max + 1e-6)
    midfft_boud    = midfft_raw    / (mid_max + 1e-6)
    treblefft_boud = treblefft_raw / (treble_max + 1e-6)



    #Do final calculationsm, ensuring they are prepared for no sound to be present
    if len(bassfft_boud) > 0:
        bassCanitadeBins = detectPeaks(bassfft_boud)
        bassCanitadeBins.sort(key=lambda i: bassfft_boud[i], reverse=True) #Sort the peaks
        final_bass = top_peaks(bassCanitadeBins,bassfft_boud,bassfreq_bound,num_peaks)
        bass_flux = calculateFlux(bassfft_boud,prev_bassfft,prevbass_flux) if prev_bassfft is not None else 0.0
    else:
        final_bass = [(0.0,0.0)] * num_peaks
        bass_flux = 0.0

    
    if len(treblefft_boud) > 0:
        trebleCanidateBins = detectPeaks(treblefft_boud)
        trebleCanidateBins.sort(key=lambda i: treblefft_boud[i], reverse=True) #Sort the peaks 
        final_treble = top_peaks(trebleCanidateBins,treblefft_boud,treblefreq_bound,num_peaks)
        treble_flux = calculateFlux(treblefft_boud,prevtreblefft, prevtreble_flux) if prevtreblefft is not None else 0.0
    else:
        final_treble = [(0.0,0.0)] * num_peaks
        treble_flux = 0.0

    
    if len(midfft_boud) > 0:
        midCanidateBins = detectPeaks(midfft_boud)
        midCanidateBins.sort(key=lambda i: midfft_boud[i], reverse=True) #Sort the peaks 
        final_mid = top_peaks(midCanidateBins,midfft_boud,midfreq_bound,num_peaks)
        mid_flux = calculateFlux(midfft_boud,prevmidfft,prevmid_flux) if prevmidfft is not None else 0.0
    else:
        final_mid = [(0.0,0.0)] * num_peaks
        mid_flux = 0.0


    
    #Update Memory
    prev_bassfft = bassfft_boud
    prevtreblefft = treblefft_boud
    prevmidfft = midfft_boud
    prevbass_flux = bass_flux
    prevtreble_flux = treble_flux
    prevmid_flux = mid_flux

    peaks = {
        "bass":final_bass,
        "treble": final_treble,
        "mid":final_mid
    }

    processed_audio.put((
        peaks,
        rms,
        np.mean(fft[(freqs >= 10) & (freqs <= 250)]) * bass_dampener,
        np.mean(fft[(freqs >= 2000) & (freqs <= 8000)] * treble_dampener),bass_flux,treble_flux,mid_flux
    ))

    return in_data, pa.paContinue

def calculateFlux(fft, prev_fft,prev_flux):
    #Flux - Determine change from one frame to another 
    diff = fft - prev_fft
    flux = np.sum(diff[diff > 0])  # positive spectral flux only
    normalized_flux = flux / (len(fft) * 0.15) #normalize the flux
    raw_flux = np.clip(normalized_flux, 0, 1)

    flux = prev_flux * 0.7 + raw_flux * 0.3 #Smooth transition between fluxes 
    return flux

def top_peaks(sorted_bins,band_fft,band_freqs, numPeaks):
     return [(band_fft[i], band_freqs[i]) for i in sorted_bins[:numPeaks]]
    
    
    

def detectPeaks(fft_values):
    candidate_bins = [
        i for i in range(2, len(fft_values) - 2)
        if fft_values[i] > fft_values[i - 1]
        and fft_values[i] > fft_values[i + 1]
        and fft_values[i] > MIN_PEAK_NORM
    ]
    return candidate_bins


def freq_to_note(freq, ref_freq=16.35): #Converts frequency to basic 12 notes 
    semitone_offset = 12 * np.log2(freq / ref_freq) 
    return int(round(semitone_offset)) % 12


def overlay_shapes(a, b, theta, w1=1.0, w2=1.0): #Overlays two shapes 
    x1, y1 = a(theta)
    x2, y2 = b(theta)
    return w1 * x1 + w2 * x2, w1 * y1 + w2 * y2



    
p = pa.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, stream_callback=callback)
stream.start_stream()

running = True
Gravity_Wells = [Gravity_Well(CENTER_X, CENTER_Y, Dot(CENTER_X, CENTER_Y, i)) for i in range(len(THETAS))] 

# --- SHAPE FUNCTIONS ---
def spiral(theta): return (rmsadj*50*((1+X_var_1)*theta)*np.cos(X_var*theta),rmsadj*50*((1+Y_var_1)*theta)*np.sin(Y_var*theta))
def rectangle(theta): return (rmsadj*50*(X_var/5)*np.arcsin(np.sin(theta)),rmsadj*50*(Y_var/5)*np.arcsin(np.cos(theta)))
def spider_web(theta):return (rmsadj*20*((1+X_var_1)*theta)*0.5*np.tan(X_var*theta),rmsadj*20*((1-Y_var_1)*theta**2)*0.8*np.sin(Y_var*theta) )
def noodle(theta):return (rmsadj*70*(2+np.sin(8*theta))*np.cos(X_var*.2*theta),rmsadj*50*(2+np.sin(8*theta))*np.sin(Y_var*theta))
def opposite_spiral(theta): return (rmsadj*50*((1+X_var_1)*theta)*np.cos(1.5*X_var*(-theta)),rmsadj*50*((1+Y_var_1)*theta)*np.sin(1.5*Y_var*(-theta)))
def eyeball(theta): return (rmsadj*50*(2+np.sin(theta))*np.cos(8*theta),rmsadj*50*(2+np.sin(theta))*np.sin(8*theta))
def flower(theta): return (rmsadj*50*(2+np.sin(16*theta))*np.cos(theta),rmsadj*50*(2+np.sin(16*theta))*np.sin(theta))
def dogbone(theta):return (rmsadj*100*(2+np.sin(8*theta))*np.cos(theta),rmsadj*70*(2+np.sin(4*theta))*np.sin(np.cos(theta)) )
def triangle_waves(theta):return (rmsadj*20*((1+X_var_1)*theta)*np.cos(X_var*theta*1.2)**-1,rmsadj*20*((1+Y_var_1)*theta)*np.cos(Y_var*theta*1.2)**-1)
def nothing(theta): return (0,0)

SHAPES = [
    spiral,           # 0  C
    flower,           # 1  C#
    noodle,           # 2  D
    eyeball,          # 3  D#
    dogbone,          # 4  E
    spiral,           # 5  F
    triangle_waves,   # 6  F#
    opposite_spiral,  # 7  G
    noodle,           # 8  G#
    dogbone,          # 9  A
    rectangle,        # 10 A#
    spider_web,        # 11 B
    nothing            #Used if there is not a second shape to overlay 
]

COLORS = [
    np.array([255, 73,   0]),    # 0  Red
    np.array([247, 143, 46]),    # 1  Red-Orange
    np.array([243, 243, 84]),    # 2  Yellow
    np.array([154, 229, 40]),    # 3  Yellow-Green
    np.array([92,  208, 91]),    # 4  Green
    np.array([0,   200, 140]),   # 5  Green-Cyan 
    np.array([3,   193, 205]),   # 6  Cyan  
    np.array([14,  120, 230]),   # 7  Cyan-Blue
    np.array([14,  16,  230]),   # 8  Blue
    np.array([146, 8,   231]),   # 9  Blue-Violet
    np.array([178, 28,  161]),   # 10 Purple
    np.array([255,255,255]), #White 
    np.array([230, 60,  160]),   # 11 Magenta 
    np.array([255,255,255]), #White
    np.array([128,255,128]), #Black 
]

color_selector = 12
prev_color_selector = 12

color = COLORS[12]
background_selector = 13 #Initial color for background (black)
prev_background_selector = 13
background_color = COLORS[background_selector]

def peak_note(peaks, idx): #Function to make sure there enough peaks 
                if len(peaks) > idx:
                    return freq_to_note(peaks[idx][1])
                return DEFAULT_NOTE #None 


shape_selector_1 = 0 #Make sure it has a value 
shape_selector_2 = 0


current_shape_1 = 12
current_shape_2 = 12
shape_hold_counter_1 = 0
shape_hold_counter_2 = 0

MIN_HOLD_FRAMES = 5

try:
    while running:
        if not processed_audio.empty(): 
            peaks, rms, bass, treble, bass_flux,treble_flux,mid_flux = processed_audio.get()
            prev_rms, prev_bass, prev_treble,prevbass_flux, prevtreble_flux, prevmid_flux, rev_peaks = rms, bass, treble,bass_flux,treble_flux,mid_flux,peaks
        else:
           peaks, rms, bass, treble,bass_flux,treble_flux,mid_flux = prev_peaks, prev_rms, prev_bass, prev_treble,prevbass_flux, prevtreble_flux, prevmid_flux
       

        #Game Loop Variables
        rmsadj = 1 + (rms * rmsmultiplier)
        

        #Controls warping of shape 
        X_var, Y_var = (10*(1-bass*b_t_multiplier)), (10*(1-treble*8))

        #Background Pulse
        background_color = background_color.astype(float)
        target_bgcolor = COLORS[background_selector].astype(float)
        diff_bg = target_bgcolor - background_color #Get the distance from current color to goal color 
        dist_bg = np.linalg.norm(diff_bg) #Get magnitude of the vector
        color_speed_bg = np.clip(dist_bg / 255.0, 0.03, 0.15) #Normalized to 0 to 1, clipped to make sure it doesnt go too fast or slow
        background_color += diff_bg * color_speed_bg #Move background color towards desired color

        target_intensity = np.clip(bass, 0, 1) #How much of the background to be showing (1 = showing, 0 = black)
        bg_fade_value += (target_intensity - bg_fade_value) * 0.1 #Smooth out transitions
        final_bg = np.clip(background_color * bg_fade_value, 0, 255).astype(int) #Clamp final color to make sure it is within the color range
        
        screen.fill(final_bg) #Fill Background


        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            # Check for the fullscreen toggle event
            else:
                if event.type == pygame.VIDEORESIZE:
                    CENTER_X   =  event.w // 2
                    CENTER_Y   = event.h // 2
            

                
                
    
        
       
        
        #Identify Loudest Peaks 
        if peaks:
            shape_selector_1  = peak_note(peaks["mid"], 0)  # First peak, selects primary shape
            if shape_selector_1 == None:
                shape_selector_1 = 12

            # if canidate_shape_1 == current_shape_1:
            #     shape_hold_counter_1 = 0
            # else:
            #     shape_hold_counter_1 += 1

            #     if shape_hold_counter_1 >= MIN_HOLD_FRAMES:
            #         current_shape_1 = canidate_shape_1
            #         shape_hold_counter_1 = 0
                
            shape_selector_1 = current_shape_1

            color_selector = peak_note(peaks["treble"], 0) #Second Peak, selects color of the shape
            if color_selector == None: #If no second peak then keep previous color
                color_selector = prev_color_selector  # 2nd

            shape_selector_2  = peak_note(peaks["mid"], 1) #Tird peak, selects secondary shape
            if shape_selector_2 == None: #If there is no secondary shape then select no shape (function)
                shape_selector_2 = 12 

            background_selector = peak_note(peaks["bass"], 0)  #Fourth peak, selects background color
            if background_selector == None:  #If there is no 4th peak then keep same background color
                background_selector = prev_background_selector  # 4th

            prev_background_selector = background_selector #Update memory
            prev_color_selector = color_selector #Update memory 

            #Smooting between color changes 
            target_color = COLORS[color_selector].astype(float)
            color = color.astype(float)
            diff = target_color - color #Get the distance from current color to goal color 
            dist = np.linalg.norm(diff) #Get magnitude of the vector
            color_speed = np.clip(dist / 255.0, 0.05, 0.35) #Normalized to 0 to 1, clipped to make sure it doesnt go too fast or slow
            color += diff * color_speed
            color = np.clip(color, 0, 255).astype(np.uint8) #Clamp to make sure it is a valid color

            
        # Compact expansion logic
        
        shape_expansion = np.clip(1.0 + (bass * 0.6) + (treble * 0.8), 1.0, 3.0)   
        print(treble_flux)
        w1 = 0.5 * bass_flux
        w2 = 1 - w1 
        
        effective_smooth = np.clip(WELL_SMOOTH + (treble_flux), 0.01, 1.0) #Better smoothing variable, snap when instrument hits but go slower when music is slower
        for i, Well in enumerate(Gravity_Wells): 
            tx, ty = overlay_shapes(SHAPES[shape_selector_1],SHAPES[shape_selector_2],THETAS[i],0.6,0.4)

            target_x = (tx * shape_expansion) + CENTER_X
            target_y = (ty * shape_expansion) + CENTER_Y
            
            nx = (1 - effective_smooth) * Well.center_x + effective_smooth * target_x #smooth motion of the wells 
            ny = (1 - effective_smooth) * Well.center_y + effective_smooth * target_y

            
            Well.update(nx, ny, screen, rms, tuple(color))

        pygame.display.flip() 

finally:    
    stream.stop_stream()
    stream.close()
    p.terminate()
    pygame.quit()