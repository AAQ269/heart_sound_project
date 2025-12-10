# src/star_pcg.py

import numpy as np
from scipy.signal import correlate, find_peaks, medfilt, butter, sosfilt
from scipy.fft import fft, ifft
import warnings
warnings.filterwarnings("ignore")


class STAR_PCG:
    """
    STAR-PCG: Self-learning Temporal-Adaptive Recognition for Phonocardiograms (PCG)
    -------------------------------------------------------------------------------
    This class implements:
        - Phase 1: Initialization (template + noise + heart period)
        - Phase 2: Adaptive Denoising for each heartbeat cycle
        - Phase 3: Template Updating + Statistics
    """

    # ==========================================================================
    # Initialization
    # ==========================================================================

    def __init__(self, fs=1000, verbose=True):
        self.fs = fs
        self.verbose = verbose

        # Learned params
        self.T_avg = None
        self.template_time = None
        self.template_freq = None
        self.noise_mean = 0
        self.noise_std = 0
        self.noise_type = "UNKNOWN"

        self.initialized = False

        # Statistics
        self.cycles_processed = 0
        self.stats = {
            "skipped": 0,
            "light": 0,
            "heavy": 0,
            "confidences": []
        }

        if verbose:
            print(f" STAR-PCG fully initialized (fs={fs})")


    # ==========================================================================
    # PHASE 1 — Initialization
    # ==========================================================================

    def initialize(self, signal, duration=5):
        """Learn template, noise type, heart period from first few seconds"""

        if self.verbose:
            print("\n" + "="*60)
            print(" PHASE 1: Initialization Started")
            print("="*60)

        init_samples = min(int(duration * self.fs), len(signal))
        sig_init = signal[:init_samples]

        # --- Step 1: Estimate Heart Period
        self.T_avg = self._estimate_heart_period(sig_init)

        if self.verbose:
            print(f" Estimated Heart Period: {self.T_avg:.3f} sec")
            print(f" HR ≈ {60/self.T_avg:.1f} BPM")

        # --- Step 2: Segment cycles
        cycles = self._segment_cycles(sig_init, self.T_avg)

        if self.verbose:
            print(f" Extracted cycles: {len(cycles)}")

        # --- Step 3: Build template
        if len(cycles) >= 2:
            self.template_time = self._build_template(cycles)
        else:
            self.template_time = sig_init[: int(self.T_avg * self.fs)]

        self.template_freq = fft(self.template_time)

        if self.verbose:
            print(f" Template built: {len(self.template_time)} samples")

        # --- Step 4: Characterize noise
        self._characterize_noise(cycles)

        if self.verbose:
            print(f" Noise Type: {self.noise_type}")
            print(f" Noise STD: {self.noise_std:.4f}")
            print(" Initialization COMPLETE\n")

        self.initialized = True


    # Heart period estimation -------------------------------------------------

    def _estimate_heart_period(self, sig):
        autocorr = correlate(sig, sig, mode="full")
        autocorr = autocorr[len(autocorr)//2:]

        min_s = int(0.3*self.fs)   # 40–200 BPM
        max_s = int(1.5*self.fs)

        max_s = min(max_s, len(autocorr))

        peaks, _ = find_peaks(autocorr[min_s:max_s], distance=min_s)

        if len(peaks) > 0:
            p = peaks[0] + min_s
            return p / self.fs

        return 0.8  # fallback


    # Segmentation ------------------------------------------------------------

    def _segment_cycles(self, sig, period):
        L = int(period * self.fs)
        if L <= 0:
            L = int(0.8 * self.fs)

        cycles = []
        for i in range(len(sig) // L):
            start, end = i*L, i*L + L
            cycles.append(sig[start:end])

        return cycles


    # Template building -------------------------------------------------------

    def _build_template(self, cycles):
        min_len = min(len(c) for c in cycles)
        aligned = [c[:min_len] for c in cycles]
        return np.mean(aligned, axis=0)


    # Noise characterization --------------------------------------------------

    def _characterize_noise(self, cycles):
        quiet = []
        for c in cycles:
            if len(c) > 20:
                q = c[int(0.7*len(c)):]
                quiet.extend(q)

        if len(quiet) == 0:
            self.noise_type = "UNKNOWN"
            self.noise_std = 0.02
            return

        quiet = np.abs(np.array(quiet))
        self.noise_mean = np.mean(quiet)
        self.noise_std = np.std(quiet)

        # Noise type classification
        if self.noise_std < 0.02:
            self.noise_type = "LOW"
        elif self.noise_std < 0.05:
            self.noise_type = "MEDIUM"
        else:
            self.noise_type = "HIGH"


    # ==========================================================================
    # PHASE 2 — Adaptive Denoising
    # ==========================================================================

    def denoise(self, signal):
        if not self.initialized:
            raise RuntimeError(" ERROR: STAR-PCG must be initialized first.")

        if self.verbose:
            print(" PHASE 2: Adaptive Denoising Started")

        cycles = self._segment_cycles(signal, self.T_avg)

        if len(cycles) == 0:
            return signal, {"total_cycles": 0}

        clean_cycles = []
        details = []

        for i, cycle in enumerate(cycles):

            # Step 1: Confidence score
            conf = self._calculate_confidence(cycle)
            self.stats["confidences"].append(conf)

            # Step 2: Adaptive decision
            if conf >= 0.80:
                clean = cycle
                mode = "SKIP"
                self.stats["skipped"] += 1

            elif conf >= 0.60:
                clean = self._light_filter(cycle)
                mode = "LIGHT"
                self.stats["light"] += 1

            else:
                clean = self._heavy_filter(cycle)
                mode = "HEAVY"
                self.stats["heavy"] += 1

            clean_cycles.append(clean)
            details.append({"cycle": i, "confidence": conf, "processing": mode})

            # Step 3: Update template every 10 good cycles
            if (i+1) % 10 == 0 and conf >= 0.75:
                self._update_template(clean)

            self.cycles_processed += 1

        # Reconstruct cleaned signal
        clean_signal = np.concatenate(clean_cycles)

        # Add remaining tail of original signal
        remaining = len(signal) - len(clean_signal)
        if remaining > 0:
            clean_signal = np.concatenate([clean_signal, signal[-remaining:]])

        if self.verbose:
            total = len(cycles)
            print(f"✓ Total cycles: {total}")
            print(f"   Skipped: {self.stats['skipped']}")
            print(f"   Light:   {self.stats['light']}")
            print(f"   Heavy:   {self.stats['heavy']}")
            print(f"   Avg Confidence: {np.mean(self.stats['confidences']):.3f}")

        return clean_signal, {"details": details, "total_cycles": len(cycles)}


    # Confidence --------------------------------------------------------------

    def _calculate_confidence(self, cycle):
        if len(cycle) == 0 or len(self.template_time) == 0:
            return 0

        L = min(len(cycle), len(self.template_time))
        c = cycle[:L]
        t = self.template_time[:L]

        num = np.sum(c * t)
        den = np.sqrt(np.sum(c**2) * np.sum(t**2) + 1e-12)

        conf = num / den
        return max(0, min(1, conf))


    # LIGHT filter ------------------------------------------------------------

    def _light_filter(self, cycle):
        if len(cycle) < 5:
            return cycle

        if self.noise_type == "LOW":
            return medfilt(cycle, kernel_size=3)

        elif self.noise_type == "MEDIUM":
            return medfilt(cycle, kernel_size=5)

        else:
            # High-pass for strong respiration/motion noise
            cutoff = 40
            sos = butter(4, cutoff, btype="highpass", fs=self.fs, output="sos")
            return sosfilt(sos, cycle)


    # HEAVY filter (Adaptive Wiener) -----------------------------------------

    def _heavy_filter(self, cycle):
        if len(cycle) == 0:
            return cycle

        fft_c = fft(cycle)
        fft_t = self.template_freq

        L = min(len(fft_c), len(fft_t))
        fft_c = fft_c[:L]
        fft_t = fft_t[:L]

        S = np.abs(fft_t)**2
        N = self.noise_std**2

        gain = S / (S + N + 1e-10)

        clean_freq = fft_c * gain
        clean = np.real(ifft(clean_freq, n=len(cycle)))

        return clean


    # Template Updating -------------------------------------------------------

    def _update_template(self, clean_cycle):
        L = min(len(clean_cycle), len(self.template_time))
        c = clean_cycle[:L]
        t = self.template_time[:L]

        # learning rate decreases over time
        age = min(self.cycles_processed / 100, 0.8)
        eta = 0.1 * (1 - age)

        self.template_time[:L] = (1-eta)*t + eta*c
        self.template_freq = fft(self.template_time)


    # ==========================================================================
    # PHASE 3 — Stats
    # ==========================================================================

    def get_stats(self):
        total = (self.stats["skipped"] +
                 self.stats["light"] +
                 self.stats["heavy"])

        if total == 0:
            return None

        energy_saved = (
            self.stats["skipped"]*100 +
            self.stats["light"]*60 +
            self.stats["heavy"]*0
        ) / total

        return {
            "total_cycles": total,
            "skipped": self.stats["skipped"],
            "light": self.stats["light"],
            "heavy": self.stats["heavy"],
            "skipped_percent": self.stats["skipped"] / total * 100,
            "light_percent": self.stats["light"] / total * 100,
            "heavy_percent": self.stats["heavy"] / total * 100,
            "avg_confidence": np.mean(self.stats["confidences"]),
            "energy_saved_percent": energy_saved
        }
