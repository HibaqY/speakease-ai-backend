"""
SpeakEase FastAPI Backend - EXACT MATCH TO TRAINING CODE
Speech and Language Development Screening for Children (Ages 2-7)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as transforms
import numpy as np
import io
import os
from typing import Optional
from datetime import datetime
import logging
import joblib
from scipy.fftpack import dct
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SpeakEase API",
    description="AI-powered speech development screening for children",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# MODEL ARCHITECTURES (EXACT MATCH TO TRAINING)
# ============================================================================

class ResidualBlock(nn.Module):
    """Residual block for MLP"""
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += residual
        out = F.relu(out)
        return out


class ImprovedMLP(nn.Module):
    """Deeper MLP with residual connections - Stage 1 Model"""
    def __init__(self, in_dim, n_classes, p=0.35):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p)
        )

        self.res_blocks = nn.Sequential(
            ResidualBlock(512, p),
            ResidualBlock(512, p)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.classifier(x)


class AttentionLayer(nn.Module):
    """Self-attention for BiLSTM outputs"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, lstm_out):
        attn_weights = self.attention(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)
        weighted = (lstm_out * attn_weights).sum(dim=1)
        return weighted


class ImprovedCNN_BiLSTM(nn.Module):
    """Enhanced CNN+BiLSTM with attention - Stage 2 Model"""
    def __init__(self, n_classes, lstm_hidden=128, lstm_layers=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.Dropout2d(0.2),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout2d(0.2),
            nn.Conv2d(256, 256, 3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=256, hidden_size=lstm_hidden, num_layers=lstm_layers,
            batch_first=True, bidirectional=True, dropout=0.3 if lstm_layers > 1 else 0
        )

        self.attention = AttentionLayer(lstm_hidden * 2)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(lstm_hidden * 2, n_classes)

    def forward(self, x):
        f = self.cnn(x)
        f = F.adaptive_avg_pool2d(f, (1, f.shape[-1]))
        f = f.squeeze(2).permute(0, 2, 1)
        lstm_out, _ = self.lstm(f)
        h = self.attention(lstm_out)
        h = self.dropout(h)
        return self.fc(h)


# ============================================================================
# FEATURE EXTRACTION (EXACT MATCH TO TRAINING)
# ============================================================================

def extract_enhanced_features(spec: np.ndarray):
    """
    Enhanced feature extraction - EXACTLY 303 dimensions
    Matches the training code's extract_enhanced_features function
    """
    n_mels, n_frames = spec.shape

    # 1. Basic statistics (64*4 = 256)
    mu  = spec.mean(axis=1)
    sig = spec.std(axis=1)

    if n_frames > 1:
        d1 = np.diff(spec, axis=1)
        d1_mu  = d1.mean(axis=1)
        d1_sig = d1.std(axis=1)
    else:
        d1_mu  = np.zeros(n_mels, dtype=np.float32)
        d1_sig = np.zeros(n_mels, dtype=np.float32)

    # 2. Temporal envelope (mean over frequency at each time)
    te = spec.mean(axis=0)
    g_mu, g_sig = te.mean(), te.std()
    p5, p25, p50, p75, p95 = np.percentile(te, [5, 25, 50, 75, 95]) if te.size > 0 else (0,)*5
    dyn = p95 - p5

    # 3. Spectral features per frame, then aggregate
    freqs = np.arange(n_mels)
    centroids, rolloffs, fluxes = [], [], []

    for t in range(n_frames):
        frame = spec[:, t]
        total = frame.sum()

        # Spectral centroid
        if total > 1e-6:
            centroid = (freqs * frame).sum() / total
        else:
            centroid = 0.0
        centroids.append(centroid)

        # Spectral rolloff (85% energy)
        if total > 1e-6:
            cumsum = np.cumsum(frame)
            rolloff_idx = np.where(cumsum >= 0.85 * total)[0]
            rolloff = rolloff_idx[0] if len(rolloff_idx) > 0 else n_mels - 1
        else:
            rolloff = 0.0
        rolloffs.append(rolloff)

        # Spectral flux
        if t > 0:
            flux = np.sum((frame - spec[:, t-1])**2)
        else:
            flux = 0.0
        fluxes.append(flux)

    centroids = np.array(centroids)
    rolloffs = np.array(rolloffs)
    fluxes = np.array(fluxes)

    # Aggregate spectral features (12 features)
    spec_feats = np.array([
        centroids.mean(), centroids.std(),
        rolloffs.mean(), rolloffs.std(),
        fluxes.mean(), fluxes.std(),
        (spec**2).sum(axis=0).mean(),
        (spec**2).sum(axis=0).std(),
        np.median(centroids),
        np.median(rolloffs),
        p25, p75
    ], dtype=np.float32)

    # 4. DCT coefficients (MFCC-style) - first 13 coefficients
    if n_frames >= 13:
        mfcc_style = dct(spec, axis=1, norm='ortho')[:, :13]
        mfcc_mu = mfcc_style.mean(axis=0)
        mfcc_sig = mfcc_style.std(axis=0)
    else:
        mfcc_mu = np.zeros(13, dtype=np.float32)
        mfcc_sig = np.zeros(13, dtype=np.float32)

    # 5. Zero-crossing rate
    zcr = ((te[:-1] * te[1:]) < 0).sum() / max(len(te)-1, 1)

    # Combine all features: 256 + 9 + 12 + 26 = 303
    feat = np.concatenate([
        mu, sig, d1_mu, d1_sig,  # 256
        np.array([g_mu, g_sig, dyn, p5, p25, p50, p75, p95, zcr], dtype=np.float32),  # 9
        spec_feats,  # 12
        mfcc_mu, mfcc_sig  # 26
    ])

    return feat.astype(np.float32)


# ============================================================================
# AUDIO PROCESSING
# ============================================================================

class AudioProcessor:
    def __init__(self, target_sr=16000, n_mels=64):
        self.target_sr = target_sr
        self.n_mels = n_mels
        self.mel_transform = transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=512,
            hop_length=256,
            n_mels=n_mels
        )
    
    def load_audio(self, audio_bytes: bytes) -> torch.Tensor:
        """Load audio from bytes and convert to target sample rate"""
        try:
            waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if necessary
            if sample_rate != self.target_sr:
                resampler = transforms.Resample(sample_rate, self.target_sr)
                waveform = resampler(waveform)
            
            return waveform
        except Exception as e:
            raise ValueError(f"Error loading audio: {str(e)}")
    
    def extract_mel_spectrogram(self, waveform: torch.Tensor) -> np.ndarray:
        """Extract mel-spectrogram (64, T) matching training preprocessing"""
        mel_spec = self.mel_transform(waveform)
        
        # Convert to dB
        mel_spec_db = transforms.AmplitudeToDB()(mel_spec)
        
        # Convert to numpy and ensure shape is (64, T)
        spec_np = mel_spec_db.squeeze().cpu().numpy()
        
        if spec_np.ndim == 1:
            spec_np = spec_np[:, None]
        
        if spec_np.shape[0] != 64:
            if spec_np.shape[1] == 64:
                spec_np = spec_np.T
        
        # Replace any non-finite values
        spec_np = np.where(np.isfinite(spec_np), spec_np, 0.0)
        
        return spec_np.astype(np.float32)
    
    def extract_stage1_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract 303-dimensional features for Stage 1"""
        mel_spec = self.extract_mel_spectrogram(waveform)
        features = extract_enhanced_features(mel_spec)
        return torch.from_numpy(features).unsqueeze(0)  # Add batch dimension
    
    def extract_stage2_features(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract and resize mel-spectrogram to (1, 1, 128, 512) for Stage 2"""
        mel_spec = self.extract_mel_spectrogram(waveform)
        
        # Normalize to [0, 1]
        mn, mx = mel_spec.min(), mel_spec.max()
        if mx - mn < 1e-6:
            mel_spec = np.zeros_like(mel_spec, dtype=np.float32)
        else:
            mel_spec = (mel_spec - mn) / (mx - mn)
        
        # Convert to tensor and resize
        x = torch.from_numpy(mel_spec)[None, None, ...]  # (1, 1, 64, T)
        x = F.interpolate(x, size=(128, 512), mode="bilinear", align_corners=False)
        
        return x


# ============================================================================
# MODEL LOADING
# ============================================================================

class ModelManager:
    def __init__(self, models_dir="models"):
        self.models_dir = models_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load models and scaler
        self.stage1_model = self._load_stage1_model()
        self.stage2_model = self._load_stage2_model()
        self.scaler = self._load_scaler()
    
    def _load_stage1_model(self):
        """Load Stage 1 MLP model"""
        model_path = os.path.join(self.models_dir, "stage1_model.pth")
        model = ImprovedMLP(in_dim=303, n_classes=2, p=0.35)
        
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"✅ Stage 1 model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load Stage 1 model: {e}. Using untrained model.")
        else:
            logger.warning(f"Stage 1 model not found at {model_path}. Using untrained model.")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _load_stage2_model(self):
        """Load Stage 2 CNN-BiLSTM model"""
        model_path = os.path.join(self.models_dir, "stage2_model.pth")
        model = ImprovedCNN_BiLSTM(n_classes=3, lstm_hidden=128, lstm_layers=2)
        
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info(f"✅ Stage 2 model loaded from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load Stage 2 model: {e}. Using untrained model.")
        else:
            logger.warning(f"Stage 2 model not found at {model_path}. Using untrained model.")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _load_scaler(self):
        """Load RobustScaler for Stage 1 features"""
        scaler_path = os.path.join(self.models_dir, "stage1_scaler.pkl")
        
        if os.path.exists(scaler_path):
            try:
                scaler = joblib.load(scaler_path)
                logger.info(f"✅ Scaler loaded from {scaler_path}")
                return scaler
            except Exception as e:
                logger.warning(f"Could not load scaler: {e}")
        else:
            logger.warning(f"Scaler not found at {scaler_path}")
        
        return None


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class AnalysisResponse(BaseModel):
    success: bool
    stage1_result: str
    stage1_confidence: float
    stage2_result: Optional[str] = None
    stage2_confidence: Optional[float] = None
    explanation: str
    needs_evaluation: bool
    timestamp: str

# ============================================================================
# AUTO-DOWNLOAD MODEL FILES (Google Drive)
# ============================================================================

def download_if_missing(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        logger.info(f"⬇️ Downloading {os.path.basename(path)} from cloud...")
        try:
            r = requests.get(url, allow_redirects=True)
            if r.status_code == 200:
                with open(path, "wb") as f:
                    f.write(r.content)
                logger.info(f"✅ Downloaded {os.path.basename(path)}")
            else:
                logger.warning(f"⚠️ Failed to download {path}, status {r.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Error downloading {path}: {e}")

# Google Drive → direct download links
STAGE1_URL = "https://drive.google.com/uc?export=download&id=195-zaraGwuV5ef-SPnksBwEu2licA94B"
STAGE2_URL = "https://drive.google.com/uc?export=download&id=1OesJaFsvw9bV9paRyKg9VwttGPZ_tBoO"
SCALER_URL = "https://drive.google.com/uc?export=download&id=13VdQVfIGozjfEYJa809RjRPvNeq3OO7q"

# Local paths used by the model loader
STAGE1_PATH = "models/stage1_model.pth"
STAGE2_PATH = "models/stage2_model.pth"
SCALER_PATH = "models/stage1_scaler.pkl"

# Download if missing
download_if_missing(STAGE1_URL, STAGE1_PATH)
download_if_missing(STAGE2_URL, STAGE2_PATH)
download_if_missing(SCALER_URL, SCALER_PATH)

logger.info("✅ Model files checked.")


# ============================================================================
# INITIALIZE MODELS
# ============================================================================

os.makedirs("models", exist_ok=True)

try:
    model_manager = ModelManager()
    audio_processor = AudioProcessor()
    logger.info("✅ Models and processors initialized successfully")
except Exception as e:
    logger.error(f"❌ Error initializing models: {e}")
    model_manager = None
    audio_processor = None


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "SpeakEase API - Speech Development Screening",
        "version": "1.0.0",
        "status": "running",
        "model": "Hybrid Deep Learning (Enhanced MLP + CNN-BiLSTM-Attention)"
    }


@app.get("/health")
async def health_check():
    models_loaded = (model_manager is not None and
                    model_manager.stage1_model is not None and
                    model_manager.stage2_model is not None)
    scaler_loaded = model_manager.scaler is not None if model_manager else False
    
    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "scaler_loaded": scaler_loaded,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/analyze/speech", response_model=AnalysisResponse)
async def analyze_speech(audio: UploadFile = File(...)):
    """
    Main endpoint for speech analysis
    Two-stage classification:
    1. TD vs Atypical (ImprovedMLP)
    2. If Atypical: DS / LT / SLI (ImprovedCNN_BiLSTM)
    """
    if not model_manager or not audio_processor:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if not audio.content_type or not audio.content_type.startswith('audio/'):
        raise HTTPException(status_code=400, detail="File must be an audio file")
    
    try:
        # Read and process audio
        audio_bytes = await audio.read()
        logger.info(f"📁 Received: {audio.filename} ({len(audio_bytes)} bytes)")
        
        waveform = audio_processor.load_audio(audio_bytes)
        logger.info(f"🎵 Audio loaded: {waveform.shape[1]} samples @ {audio_processor.target_sr}Hz")
        
        # ===== STAGE 1: TD vs Atypical =====
        stage1_features = audio_processor.extract_stage1_features(waveform)
        logger.info(f"📊 Stage 1 features: {stage1_features.shape}")
        
        # Apply scaler if available
        if model_manager.scaler:
            stage1_features_np = stage1_features.cpu().numpy()
            stage1_features_scaled = model_manager.scaler.transform(stage1_features_np)
            stage1_features = torch.from_numpy(stage1_features_scaled).float()
        
        stage1_features = stage1_features.to(model_manager.device)
        
        with torch.no_grad():
            stage1_output = model_manager.stage1_model(stage1_features)
            stage1_probs = torch.softmax(stage1_output, dim=1)
            stage1_pred = torch.argmax(stage1_probs, dim=1).item()
            stage1_confidence = stage1_probs[0][stage1_pred].item()
        
        stage1_labels = ["Typical Development (TD)", "Atypical Development"]
        stage1_result = stage1_labels[stage1_pred]
        
        logger.info(f"✅ Stage 1: {stage1_result} (conf: {stage1_confidence:.3f})")
        
        # Confidence threshold for evaluation (matching training: 0.65)
        needs_evaluation = stage1_confidence < 0.70
        
        # ===== STAGE 2: If Atypical, classify specific condition =====
        stage2_result = None
        stage2_confidence = None
        
        if stage1_pred == 1:  # Atypical
            stage2_features = audio_processor.extract_stage2_features(waveform)
            logger.info(f"📊 Stage 2 features: {stage2_features.shape}")
            
            stage2_features = stage2_features.to(model_manager.device)
            
            with torch.no_grad():
                stage2_output = model_manager.stage2_model(stage2_features)
                stage2_probs = torch.softmax(stage2_output, dim=1)
                stage2_pred = torch.argmax(stage2_probs, dim=1).item()
                stage2_confidence = stage2_probs[0][stage2_pred].item()
            
            stage2_labels = ["Down Syndrome", "Late Talker", "Speech/Language Disorder (SLI)"]
            stage2_result = stage2_labels[stage2_pred]
            
            logger.info(f"✅ Stage 2: {stage2_result} (conf: {stage2_confidence:.3f})")
            
            # Update evaluation flag (matching training: 0.65)
            if stage2_confidence < 0.65:
                needs_evaluation = True
        
        # Generate explanation
        explanation = _generate_explanation(
            stage1_result, stage1_confidence,
            stage2_result, stage2_confidence,
            needs_evaluation
        )
        
        return AnalysisResponse(
            success=True,
            stage1_result=stage1_result,
            stage1_confidence=round(stage1_confidence, 4),
            stage2_result=stage2_result,
            stage2_confidence=round(stage2_confidence, 4) if stage2_confidence else None,
            explanation=explanation,
            needs_evaluation=needs_evaluation,
            timestamp=datetime.now().isoformat()
        )
    
    except ValueError as e:
        logger.error(f"❌ Value error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing speech: {str(e)}")


def _generate_explanation(stage1_result: str, stage1_conf: float,
                          stage2_result: Optional[str], stage2_conf: Optional[float],
                          needs_eval: bool) -> str:
    """Generate human-readable explanation"""
    
    explanation = f"Analysis indicates {stage1_result} "
    explanation += f"(confidence: {stage1_conf*100:.1f}%). "
    
    if stage2_result:
        explanation += f"\n\nSpecific pattern detected: {stage2_result} "
        explanation += f"(confidence: {stage2_conf*100:.1f}%). "
    
    if needs_eval:
        explanation += "\n\n⚠️ Confidence is below threshold. We recommend consulting with a speech-language pathologist for comprehensive evaluation."
    else:
        if "Typical" in stage1_result:
            explanation += "\n\n✅ Speech patterns appear to be within typical developmental range for age."
        else:
            explanation += "\n\n📋 This screening suggests patterns that may benefit from professional evaluation."
    
    explanation += "\n\n⚕️ Important: This is a screening tool, not a diagnostic instrument. Always consult qualified healthcare professionals for proper assessment."
    
    return explanation


@app.get("/api/model/info")
async def get_model_info():
    """Get information about the models"""
    return {
        "model_name": "SpeakEase Hybrid Deep Learning System",
        "version": "1.0.0",
        "stage1": {
            "architecture": "ImprovedMLP (Residual Blocks)",
            "classes": ["TD", "Atypical"],
            "input_features": 303,
            "feature_types": [
                "Basic statistics (256)",
                "Temporal envelope (9)",
                "Spectral features (12)",
                "DCT coefficients (26)"
            ]
        },
        "stage2": {
            "architecture": "CNN + BiLSTM + Attention",
            "classes": ["Down Syndrome", "Late Talker", "SLI"],
            "input_shape": [1, 1, 128, 512]
        },
        "preprocessing": {
            "sample_rate": "16 kHz",
            "mel_bands": 64,
            "scaler": "RobustScaler",
            "confidence_threshold": 0.65
        },
        "target_age_range": "2-7 years"
    }


if __name__ == "__main__":
    print("=" * 60)
    print("🎯 SpeakEase Backend - Starting Server")
    print("=" * 60)
    print("Model Architecture: Hybrid Deep Learning")
    print("Stage 1: ImprovedMLP with Residual Connections")
    print("Stage 2: CNN + BiLSTM + Attention")
    print("Feature Extraction: 303-dimensional enhanced features")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
