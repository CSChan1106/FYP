from flask import Flask, request, jsonify, session, send_from_directory, render_template, redirect, url_for, flash, Response
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import mne
import numpy as np
import joblib
import tempfile
import os
import json
from flask_cors import CORS
from tensorflow.keras.models import load_model
from datetime import datetime
from flask import send_from_directory
import http.client
import json
import re
import uuid

app = Flask(__name__)
CORS(app, supports_credentials=True)

app.config['SECRET_KEY'] = 'your_secret_key'  
app.config['SQLALCHEMY_DATABASE_URI'] ='postgresql://postgres:admin@localhost/eeg_db' 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_COOKIE_SAMESITE'] = "None"
app.config['SESSION_COOKIE_SECURE'] = True

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

##############################################################################
# Def Section
##############################################################################


def format_response(ai_reply):
    # Check for numbered lists
    numbered_list_pattern = re.compile(r'^(\d+)\.\s(.*)$', re.MULTILINE)
    
    # Check for bullet points (assuming a dash or asterisk)
    bullet_list_pattern = re.compile(r'^[•-]\s(.*)$', re.MULTILINE)

    formatted_reply = []
    seen_numbers = set()  # To keep track of seen numbers

    # Split the reply into lines
    lines = ai_reply.strip().split('\n')

    for line in lines:
        # Check for numbered list items
        match = numbered_list_pattern.match(line)
        if match:
            number = match.group(1)  # Extract the number
            content = match.group(2)  # Extract the content
            
            if number not in seen_numbers:  # Check if the number has been seen
                seen_numbers.add(number)  # Mark this number as seen
                formatted_reply.append(f"<li>{content}</li>")
        
        # Check for bullet point items
        elif bullet_list_pattern.match(line):
            formatted_reply.append(f"<li>{line[2:]}</li>")  # Skip the bullet character
        else:
            # Add any non-list text as a paragraph
            formatted_reply.append(f"<p>{line}</p>")

    # Wrap the list items in <ol> or <ul> tags
    if any(numbered_list_pattern.match(line) for line in lines):
        return "<ol>" + "".join(formatted_reply) + "</ol>"
    else:
        return "<ul>" + "".join(formatted_reply) + "</ul>"


##############################################################################
# Load pre-trained models and scalers
##############################################################################

# Base directory where your model and scaler files are located
BASE_DIR =""

# Load models
AD_MODEL_PATH = os.path.join(BASE_DIR, "model", "ad.h5")  
AD_SCALER_PATH = os.path.join(BASE_DIR, "model", "ad.pkl")  
ad_model = load_model(AD_MODEL_PATH)
ad_scaler = joblib.load(AD_SCALER_PATH)

AXT_MODEL_PATH = os.path.join(BASE_DIR, "model", "anxiety.h5")  
AXT_SCALER_PATH = os.path.join(BASE_DIR, "model", "anxiety.pkl")  
axt_model = load_model(AXT_MODEL_PATH)
axt_scaler = joblib.load(AXT_SCALER_PATH)

FTD_MODEL_PATH = os.path.join(BASE_DIR, "model", "ftd.h5")  
FTD_SCALER_PATH = os.path.join(BASE_DIR, "model", "ftd.pkl")  
ftd_model = load_model(FTD_MODEL_PATH)
ftd_scaler = joblib.load(FTD_SCALER_PATH)

PKS_MODEL_PATH = os.path.join(BASE_DIR, "model", "pk.h5")  
PKS_SCALER_PATH = os.path.join(BASE_DIR, "model", "pk.pkl")  
pks_model = load_model(PKS_MODEL_PATH)
pks_scaler = joblib.load(PKS_SCALER_PATH)
print("AD Model Input Shape:", ad_model.input_shape) 
print("AXT Model Input Shape:", axt_model.input_shape)  
print("FTD Model Input Shape:", ftd_model.input_shape)   
print("PKS Model Input Shape:", pks_model.input_shape)  
print("All models and scalers loaded successfully!")

##############################################################################
# Database Models
##############################################################################

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(256), nullable=False)
    name = db.Column(db.String(150), nullable=True)  # New field for name
    age = db.Column(db.Integer, nullable=True)      # New field for age
    icon = db.Column(db.LargeBinary, nullable=True)
    icon_mimetype = db.Column(db.String(128), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Signal_History(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.now())
    signal_data = db.Column(db.Text, nullable=False)
    sampling_rate = db.Column(db.Float, nullable=False)
    prediction_result = db.Column(db.Text, nullable=True)
    statistical_summary = db.Column(db.Text, nullable=True) 
    
class ChatSummary(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    record_ids = db.Column(db.Text, nullable=False)  # JSON list of record IDs used for the summary
    summary_text = db.Column(db.Text, nullable=False)  # ChatGPT-generated summary text
    timestamp = db.Column(db.DateTime, default=db.func.now())


class ChatHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    conversation = db.Column(db.Text, nullable=False)  # JSON string representing a list of messages
    updated_at = db.Column(db.DateTime, default=db.func.now(), onupdate=db.func.now())
    
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@login_manager.unauthorized_handler
def unauthorized():
    return jsonify({'error': 'Unauthorized access, please log in'}), 401

##############################################################################
# Helper Functions
##############################################################################

def process_pks_ad_ftd(raw,scaler):
    
    # Set average reference (as used in the paper)
    raw.set_eeg_reference()
    
    # Crop data for homogeneity (here, 0 to 100 s; adjust if needed)
    raw.crop(tmin=0, tmax=120)
    
    # Bandpass filter to remove noise (0.5–45 Hz, consistent with the paper)
    raw.filter(l_freq=0.5, h_freq=45, fir_design='firwin')
    
    # Create fixed-length epochs of 2 s duration with 1 s overlap
    epochs = mne.make_fixed_length_epochs(raw, duration=2, overlap=1, preload=True)
    
    # Extract epoch data into a NumPy array (shape: n_trials, n_channels, n_timepoints)
    epochs_array = epochs.get_data()
    
    # Resample each epoch to a target sampling rate of 128 Hz.
    original_sampling_rate = raw.info['sfreq']  # e.g., 500 Hz
    target_sampling_rate = 128
    epochs_array = mne.filter.resample(epochs_array, up=target_sampling_rate, down=int(original_sampling_rate))
 
    # Standardize each epoch (for each PCA component across time)
    epochs_array = (epochs_array - epochs_array.mean(axis=-1, keepdims=True)) / epochs_array.std(axis=-1, keepdims=True)
    epochs_array= np.expand_dims(epochs_array, axis=-1)
    new_shape = epochs_array.shape
    epochs_array = scaler.transform(epochs_array.reshape(-1, new_shape[-1])).reshape(new_shape)

    return epochs_array

def preprocess_raw_with_channel_selection(temp_filename):
    raw = mne.io.read_raw_eeglab(temp_filename, preload=True)
    target_channels = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
                       "C3", "Cz", "C4", "P3", "P4", "O1", "O2"]
    montage = mne.channels.make_standard_montage('standard_1020')
    target_positions = {ch: montage.get_positions()['ch_pos'][ch] for ch in target_channels if ch in montage.ch_names}

    if raw.get_montage() is None:
        raw.set_montage(montage)

    raw_positions = {ch: raw.info['chs'][idx]['loc'][:3] for idx, ch in enumerate(raw.info['ch_names']) if np.linalg.norm(raw.info['chs'][idx]['loc'][:3]) > 0}

    chosen_channels = []
    for target in target_channels:
        if target in raw.info['ch_names']:
            chosen_channels.append(target)
        elif target in target_positions:
            t_pos = np.array(target_positions[target])
            closest_channel = min(raw_positions.keys(), key=lambda ch: np.linalg.norm(np.array(raw_positions[ch]) - t_pos))
            chosen_channels.append(closest_channel)

    if len(chosen_channels) < 4:
        chosen_channels += [ch for ch in raw.info['ch_names'] if ch not in chosen_channels][:4-len(chosen_channels)]

    print("Chosen channels:", chosen_channels)
    raw.pick_channels(chosen_channels)
    return raw

def process_axt(raw, scaler, desired_feature_length=750):
 

    data, times = raw.get_data(return_times=True)
    n_channels, n_samples = data.shape
    print(f"Data shape after channel picking: {data.shape}")
    
    features_per_channel = desired_feature_length // n_channels   # integer division
    print(f"Resampling each of {n_channels} channels to {features_per_channel} points each.")
    
    resampled_features = []
    for ch_data in data:
        # Create a new time base to interpolate onto.
        old_indices = np.linspace(0, 1, num=n_samples)
        new_indices = np.linspace(0, 1, num=features_per_channel)
        resampled = np.interp(new_indices, old_indices, ch_data)
        resampled_features.append(resampled)
    
    # Concatenate the features from all channels.
    features = np.concatenate(resampled_features)  # shape (n_channels * features_per_channel, )
    print(f"Concatenated feature vector length: {len(features)}")
    
    # ---------------------------
    # Adjust the length of the concatenated features to match desired_feature_length.
    # ---------------------------
    if len(features) < desired_feature_length:
        # Pad with zeros if the length is too short.
        features = np.pad(features, (0, desired_feature_length - len(features)), mode='constant')
        print(f"Padded feature vector to length {desired_feature_length}.")
    elif len(features) > desired_feature_length:
        # Trim the vector if it is too long.
        features = features[:desired_feature_length]
        print(f"Trimmed feature vector to length {desired_feature_length}.")
    
    # ---------------------------
    # Standardize the feature vector using the saved scaler.
    # ---------------------------
    features = features.reshape(1, -1)  # reshape to (1, desired_feature_length)
    features = scaler.transform(features)
    print("Data processing complete. Feature vector is ready for prediction.")
    return features


##############################################################################
# Routes
##############################################################################

@app.route('/register', methods=['POST'])
def register():
    data = request.json
    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already exists'}), 400

    hashed_password = bcrypt.generate_password_hash(data['password']).decode('utf-8')
    new_user = User(
        email=data['email'],
        password=hashed_password,
        name=data.get('username'),
        age=data.get('age')
    )

    db.session.add(new_user)
    db.session.commit()
    return jsonify({'message': 'User registered successfully'}), 201

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        user = User.query.filter_by(email=email).first()
        
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return jsonify({'message': 'Login successful', 'redirect': url_for('test_api')})
        return jsonify({'error': 'Invalid credentials'}), 401
    return jsonify({'message': 'Logged in successfully'})


@app.route('/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    flash('Logged out successfully', 'success')
    return jsonify({'message': 'Logged out successfully', 'redirect': url_for('index')})

@app.route('/upload', methods=['POST'])
@login_required

def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No .set file provided'}), 400

    file = request.files['file']
    fdt_file = request.files.get('fdt_file')  # Optional .fdt file

    if file.filename == '':
        return jsonify({'error': 'No selected .set file'}), 400

    if not file.filename.endswith('.set'):
        return jsonify({'error': 'Invalid file format. Only .set files are allowed'}), 400

    temp_set_filename = None
    temp_fdt_filename = None

    try:
        unique_base = str(uuid.uuid4())
        # Define the temp directory in the same directory as this file
        temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp")
        os.makedirs(temp_dir, exist_ok=True)

        # Save the .set file
        temp_set_filename = os.path.join(temp_dir, f"{unique_base}.set")
        with open(temp_set_filename, "wb") as temp_set_file:
            temp_set_file.write(file.read())

        # Save the accompanying .fdt file if provided
        if fdt_file and fdt_file.filename.endswith('.fdt'):
            temp_fdt_filename = os.path.join(temp_dir, f"{unique_base}.fdt")
            with open(temp_fdt_filename, "wb") as temp_fdt_file:
                temp_fdt_file.write(fdt_file.read())

        # Process the raw data using our channel-selection routine.
        raw = preprocess_raw_with_channel_selection(temp_set_filename)

        # --- Enhanced Statistical Summary ---
        eeg_data = raw.get_data()  # shape: (channels, samples)
        duration_sec = eeg_data.shape[1] / raw.info['sfreq']
        original_sampling_rate = raw.info['sfreq']
        target_sampling_rate = 128  # The rate used for resampling in processing
        num_channels = len(raw.ch_names)
        channels = raw.ch_names

        # Global statistics across all channels
        global_mean = float(np.mean(eeg_data))
        global_std = float(np.std(eeg_data))
        global_min = float(np.min(eeg_data))
        global_max = float(np.max(eeg_data))
        global_median = float(np.median(eeg_data))

        # Per-channel statistics
        per_channel_stats = {}
        for i, ch in enumerate(channels):
            ch_data = eeg_data[i]
            per_channel_stats[ch] = {
                "mean": float(np.mean(ch_data)),
                "std": float(np.std(ch_data)),
                "min": float(np.min(ch_data)),
                "max": float(np.max(ch_data)),
                "median": float(np.median(ch_data))
            }

        # Epoch information: using fixed-length epochs (2 sec duration with 1 sec overlap)
        epoch_duration = 2  # seconds
        epoch_overlap = 1   # second
        epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, overlap=epoch_overlap, preload=True)
        num_epochs = len(epochs)

        stat_summary = {
            "duration_sec": duration_sec,
            "original_sampling_rate": original_sampling_rate,
            "processed_sampling_rate": target_sampling_rate,
            "num_channels": num_channels,
            "channels": channels,
            "global_stats": {
                "mean": global_mean,
                "std": global_std,
                "min": global_min,
                "max": global_max,
                "median": global_median
            },
            "per_channel_stats": per_channel_stats,
            "epoch_info": {
                "epoch_duration": epoch_duration,
                "epoch_overlap": epoch_overlap,
                "num_epochs": num_epochs
            }
        }
        stat_summary_str = json.dumps(stat_summary)
        # --- End of Enhanced Statistical Summary ---

        # Process data for each model.
        print("**********************************Processing data for AXT model********************************")
        data_axt = process_axt(raw, axt_scaler,750)
        prediction_axt = axt_model.predict(data_axt)
        print("AXT prediction shape:", prediction_axt.shape)
        print("AXT prediction:", prediction_axt)
        prediction_axt_mean = float(prediction_axt[0][0])    

        
        data_ad = process_pks_ad_ftd(raw, ad_scaler)
        prediction_ad = ad_model.predict(data_ad)
        prediction_ad_mean = float(np.mean(prediction_ad))

        data_ftd = process_pks_ad_ftd(raw, ftd_scaler)
        prediction_ftd = ftd_model.predict(data_ftd)
        prediction_ftd_mean = float(np.mean(prediction_ftd))

        data_pks = process_pks_ad_ftd(raw, pks_scaler)
        prediction_pks = pks_model.predict(data_pks)
        prediction_pks_mean = float(np.mean(prediction_pks))
        
        # Convert the full EEG signal to JSON for storage/plotting.
        timestamps = np.arange(eeg_data.shape[1]).tolist()
        eeg_json = {
            "timestamps": timestamps,
            "signals": {ch: eeg_data[i].tolist() for i, ch in enumerate(channels)}
        }
        eeg_json_str = json.dumps(eeg_json)
        sampling_rate = raw.info['sfreq']

        predictions = {
            "AD": prediction_ad_mean,
            "FTD": prediction_ftd_mean,
            "AXT": prediction_axt_mean,
            "PKS": prediction_pks_mean
        }

        # Create and store the signal record with the statistical summary.
        signal_record = Signal_History(
            user_id=current_user.id,
            signal_data=eeg_json_str,
            sampling_rate=sampling_rate,
            prediction_result=json.dumps(predictions),
            statistical_summary=stat_summary_str
        )
        db.session.add(signal_record)
        db.session.commit()
        
       
        return jsonify({
            'predictions': predictions,
            'record_id': signal_record.id,
            'statistical_summary': stat_summary  # Optionally return this to the frontend
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # Delete the temporary files after processing.
        if temp_set_filename and os.path.exists(temp_set_filename):
            os.remove(temp_set_filename)
        if temp_fdt_filename and os.path.exists(temp_fdt_filename):
            os.remove(temp_fdt_filename)

@app.route('/signal_history/metadata', methods=['GET'])
@login_required
def get_signal_history_metadata():
    records = Signal_History.query.filter_by(user_id=current_user.id).order_by(Signal_History.timestamp).all()
    metadata = [{'id': record.id, 'timestamp': record.timestamp.isoformat()} for record in records]
    return jsonify({'signal_history_metadata': metadata})

@app.route('/signal_history/latest', methods=['GET'])
@login_required
def get_latest_signal_history():
    record = Signal_History.query.filter_by(user_id=current_user.id).order_by(Signal_History.timestamp.desc()).first()
    if record:
        eeg_json = json.loads(record.signal_data)
        return jsonify({
            'id': record.id,
            'timestamp': record.timestamp.isoformat(),
            'eeg_data': eeg_json,
            'sampling_rate': record.sampling_rate,
            'prediction': json.loads(record.prediction_result)
        })
    else:
        return jsonify({'error': 'No signal history found.'}), 404

@app.route('/signal_history/<int:record_id>', methods=['GET'])
@login_required
def get_signal_history_by_id(record_id):
    record = Signal_History.query.filter_by(user_id=current_user.id, id=record_id).first()
    if record:
        eeg_json = json.loads(record.signal_data)
        return jsonify({
            'id': record.id,
            'timestamp': record.timestamp.isoformat(),
            'eeg_data': eeg_json,
            'sampling_rate': record.sampling_rate,
            'prediction': json.loads(record.prediction_result)
        })
    else:
        return jsonify({'error': 'Record not found.'}), 404

@app.route('/signal_history/summary', methods=['GET'])
@login_required
def get_summary():
    records = Signal_History.query.filter_by(user_id=current_user.id).all()
    if not records:
         return jsonify({'error': 'No records found'}), 404

    n_points_signal = 1000
    signals_resampled = []
    for record in records:
         data = json.loads(record.signal_data)
         avg_signal = np.mean(np.array(list(data['signals'].values())), axis=0)
         n = len(avg_signal)
         t_orig = np.linspace(0, 1, n)
         t_common = np.linspace(0, 1, n_points_signal)
         resampled = np.interp(t_common, t_orig, avg_signal)
         signals_resampled.append(resampled)
    average_signal = np.mean(signals_resampled, axis=0).tolist()

    n_points_band = 100
    bands = {
         "Delta": (0.5, 4),
         "Theta": (4, 8),
         "Alpha": (8, 13),
         "Beta": (13, 30),
         "Gamma": (30, 45)
    }
    def compute_band_power_series(signal, sr, window_duration=1.0):
         window_size = int(sr * window_duration)
         if window_size < 1:
             window_size = 1
         n_windows = len(signal) // window_size
         band_series = {band: [] for band in bands}
         for i in range(n_windows):
             segment = signal[i*window_size:(i+1)*window_size]
             fft = np.fft.rfft(segment)
             freqs = np.fft.rfftfreq(len(segment), d=1/sr)
             power = np.abs(fft)**2
             for band, (f_low, f_high) in bands.items():
                 idx = np.where((freqs >= f_low) & (freqs < f_high))[0]
                 band_power = np.mean(power[idx]) if len(idx) > 0 else 0
                 band_series[band].append(band_power)
         return band_series, n_windows

    band_series_list = {band: [] for band in bands}
    for record in records:
         data = json.loads(record.signal_data)
         avg_signal = np.mean(np.array(list(data['signals'].values())), axis=0)
         sr = record.sampling_rate
         series, n_windows = compute_band_power_series(avg_signal, sr, window_duration=1.0)
         if n_windows > 1:
             t_orig = np.linspace(0, 1, n_windows)
         else:
             t_orig = np.array([0.5])
         t_common = np.linspace(0, 1, n_points_band)
         for band in bands:
             if n_windows > 1:
                 resampled_series = np.interp(t_common, t_orig, series[band])
             else:
                 resampled_series = np.full(n_points_band, series[band][0])
             band_series_list[band].append(resampled_series)
    average_band_series = {}
    for band in bands:
         if band_series_list[band]:
             average_band_series[band] = np.mean(band_series_list[band], axis=0).tolist()
         else:
             average_band_series[band] = [0] * n_points_band

    summary = {
         "summary_signal": {
             "normalized_time": np.linspace(0, 1, n_points_signal).tolist(),
             "average_signal": average_signal
         },
         "summary_band": {
             "normalized_time": np.linspace(0, 1, n_points_band).tolist(),
             "bands": average_band_series
         }
    }
    return jsonify(summary)
@app.route('/generate_summary', methods=['POST'])
@login_required
def generate_summary():
    data = request.get_json()
    record_ids = data.get('record_ids')
    if not record_ids:
        # Automatically select up to 5 latest records if no record_ids are provided.
        records = Signal_History.query.filter_by(user_id=current_user.id)\
                    .order_by(Signal_History.timestamp.desc()).limit(5).all()
        # Reverse the list to have them in ascending order (optional)
        records = list(reversed(records))
    else:
        if not isinstance(record_ids, list) or not (1 <= len(record_ids) <= 5):
            return jsonify({'error': 'Please select between 1 to 5 records.'}), 400
        records = Signal_History.query.filter(Signal_History.user_id == current_user.id, Signal_History.id.in_(record_ids)).all()
    
    if not records:
        return jsonify({'error': 'No records found for summary generation.'}), 404

    # Build a prompt with details from each record.
    prompt_parts = []
    for rec in records:
        try:
            stat_summary = json.loads(rec.statistical_summary) if rec.statistical_summary else {}
        except Exception as e:
            stat_summary = {}
        predictions = json.loads(rec.prediction_result) if rec.prediction_result else {}
        prompt_parts.append(
            f"Record ID: {rec.id}, Timestamp: {rec.timestamp.isoformat()}, "
            f"Duration: {stat_summary.get('duration_sec', 'N/A')} sec, "
            f"Channels: {stat_summary.get('num_channels', 'N/A')}, "
            f"Global Mean: {stat_summary.get('global_stats', {}).get('mean', 'N/A')}, "
            f"Predictions: {predictions}"
        )
    records_info = "\n".join(prompt_parts)

    # Include basic user profile info.
    user_profile = f"User Age: {current_user.age if current_user.age else 'N/A'}, User Name: {current_user.name if current_user.name else 'N/A'}"

    # Construct the final prompt for ChatGPT.
    final_prompt = (
        "<!!HIDE>" 
        "You are a medical assistant specialized in brain health and EEG analysis. "
        "Based on the following EEG record details, provide a concise summary that includes the following sections in this exact order, bold the section title: "
        "1. Summary Overview\n\n"
        "2. You may suffer from (statistics fetch from your latest 5 upload EEG records): (key findings in percentage, using bullet points) "
        "3. Personalized Health Tips (4-5 actionable tips e.g. Brain exercise recommendations, life styles etc.), "
        "and 4. Conclusion. Keep the summary around 200-250 words."
        "Ensure the format is consistent. \n\n"
        "The 5 latest EEG Record Details:\n"
        f"{records_info}\n\n"
        "User Profile:\n"
        f"{user_profile}\n\n"
        "Generate the summary with the specified format.Remember to separate the parts. Focus more on the Personalized Health Tips part. Do not include 'Consultation with a Specialist' in Personalized Health tips and please add a disclaimer at the end:'Disclaimer:This is just for reference, and users should consult a professional for advice if necessary.'"
    )

    try:
        payload = json.dumps({
            "model": "gpt-4o",  # Replace with your model name if needed
            "messages": [
                {"role": "system", "content": "You are a medical assistant specialized in brain health."},
                {"role": "user", "content": final_prompt}
            ],
            "max_tokens": 500,  # Increase if needed
            "temperature": 0.5,
            "stream": False
        })

        headers = {
            'Authorization': 'Bearer sk-jqmhNrC6MgA7pxLP80E880095eDa4029AfB1Ef37E15bDa1c',  # Replace with your actual API key
            'Content-Type': 'application/json'
        }

        conn = http.client.HTTPSConnection("api.gpt.ge")
        conn.request("POST", "/v1/chat/completions", payload, headers)
        res = conn.getresponse()
        data_response = res.read()
        response_data = json.loads(data_response.decode("utf-8"))
        ai_reply = response_data.get('choices')[0].get('message').get('content', '')
    except Exception as e:
        return jsonify({'error': f'Error generating summary: {str(e)}'}), 500

    # Save the generated summary in the ChatSummary table.
    summary_record = ChatSummary(
        user_id=current_user.id,
        record_ids=json.dumps([rec.id for rec in records]),
        summary_text=ai_reply
    )
    db.session.add(summary_record)
    db.session.commit()

    return jsonify({'summary': ai_reply})

@app.route('/past_summaries', methods=['GET'])
@login_required
def get_past_summaries():
    summaries = ChatSummary.query.filter_by(user_id=current_user.id).order_by(ChatSummary.timestamp.desc()).all()
    summaries_list = []
    for summary in summaries:
        summaries_list.append({
            "id": summary.id,
            "record_ids": json.loads(summary.record_ids),
            "timestamp": summary.timestamp.isoformat(),
            # Optionally include a snippet of the summary text
            "summary": summary.summary_text
        })
    return jsonify({'past_summaries': summaries_list})

@app.route('/past_summaries/<int:summary_id>', methods=['GET'])
@login_required
def get_past_summary(summary_id):
    summary = ChatSummary.query.filter_by(user_id=current_user.id, id=summary_id).first()
    if not summary:
        return jsonify({'error': 'Summary not found.'}), 404
    return jsonify({'summary': summary.summary_text})

@app.route('/change_icon', methods=['POST'])
@login_required
def change_icon():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Debug: Print file details
    print("Received file:", file.filename)
    print("File content type:", file.content_type)

    # Read the file data and its MIME type
    file_data = file.read()
    mimetype = file.content_type

    # Store the binary data and mimetype in the database.
    current_user.icon = file_data
    current_user.icon_mimetype = mimetype
    db.session.commit()

    # Return a success message. Clients can retrieve the icon via the /user_icon/<user_id> endpoint.
    return jsonify({'message': 'Icon updated successfully'})

# New route: Serve user icon directly from the database.
@app.route('/user_icon/<int:user_id>')
def get_user_icon(user_id):
    user = User.query.get(user_id)
    if user and user.icon:
        return Response(user.icon, mimetype=user.icon_mimetype)
    else:
        return '', 404


@app.route('/update_name', methods=['POST'])
@login_required
def update_name():
    data = request.json
    new_name = data.get('name')
    
    if not new_name:
        return jsonify({'error': 'Name is required'}), 400

    try:
        print(f"Updating name for user {current_user.id} to {new_name}")
        current_user.name = new_name
        db.session.commit()
        return jsonify({'message': 'Name updated successfully', 'name': current_user.name})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/update_age', methods=['POST'])
@login_required
def update_age():
    data = request.json
    new_age = data.get('age')

    # Check if new_age is provided and is a valid integer
    if new_age is None or not isinstance(new_age, (int, str)) or (isinstance(new_age, str) and not new_age.isdigit()):
        return jsonify({'error': 'Valid age is required'}), 400

    # Convert to integer
    new_age = int(new_age)

    if new_age < 0:  # Ensure age is not negative
        return jsonify({'error': 'Age cannot be negative'}), 400

    try:
        current_user.age = new_age
        db.session.commit()
        return jsonify({'message': 'Age updated successfully'})
    except Exception as e:
        db.session.rollback()  # Rollback if there's an error
        return jsonify({'error': 'Failed to update age'}), 500

@app.route('/change_password', methods=['POST'])
@login_required
def change_password():
    data = request.json
    old_password = data.get('old_password')
    new_password = data.get('new_password')

    # Validate input
    if not old_password or not new_password:
        return jsonify({'error': 'Both old and new passwords are required'}), 400

    # Verify the old password
    if not check_password_hash(current_user.password, old_password):
        return jsonify({'error': 'Old password is incorrect'}), 400

    # Update the password with the new password
    hashed_password = bcrypt.generate_password_hash(new_password).decode('utf-8')
    current_user.password = hashed_password
    db.session.commit()

    return jsonify({'message': 'Password updated successfully'})


@app.route('/api/chat', methods=['POST'])
@login_required
def chat():
    user_message = request.json.get('message')
    if not user_message:
        return jsonify({'error': 'Message is required'}), 400

    # Retrieve the latest 5 statistical summaries from Signal_History
    latest_records = Signal_History.query.filter_by(user_id=current_user.id)\
                        .order_by(Signal_History.timestamp.desc()).limit(5).all()
    summaries = []
    for rec in reversed(latest_records):  # Reverse to maintain chronological order
        try:
            stat = json.loads(rec.statistical_summary) if rec.statistical_summary else {}
        except Exception as e:
            stat = {}
        summaries.append(
            f"Latest Record: {rec.id}, Timestamp: {rec.timestamp.isoformat()}, "
            f"Duration: {stat.get('duration_sec', 'N/A')} sec, Channels: {stat.get('num_channels', 'N/A')}, "
            f"Global Mean: {stat.get('global_stats', {}).get('mean', 'N/A')}"
        )
    context_summary = "\n".join(summaries)

    # Retrieve or initialize conversation history for this user.
    chat_history_record = ChatHistory.query.filter_by(user_id=current_user.id).first()
    if chat_history_record:
        try:
            conversation = json.loads(chat_history_record.conversation)
        except:
            conversation = []
    else:
        conversation = []

    # Update the system context message with the latest summaries.
    context_message = {
        "role": "system",
        "content": (
            " <!!HIDE> You are an AI doctor specialized in brain health. "
            f"Here are the latest five EEG statistical summaries for context:\n{context_summary}."
        )
    }

    # Update or insert the system context message.
    system_message_index = next((idx for idx, msg in enumerate(conversation) if msg.get("role") == "system"), None)
    if system_message_index is not None:
        conversation[system_message_index] = context_message
    else:
        conversation.insert(0, context_message)

    # Append the new user message.
    conversation.append({
        "role": "user",
        "content": user_message
    })

    # Build the payload for the ChatGPT API call using the conversation history.
    payload = json.dumps({
        "model": "gpt-4o",
        "messages": conversation,
        "max_tokens": 500,
        "temperature": 0.5,
        "stream": False
    })

    headers = {
        'Authorization': 'Bearer sk-jqmhNrC6MgA7pxLP80E880095eDa4029AfB1Ef37E15bDa1c',  # Use your API key securely.
        'Content-Type': 'application/json'
    }

    try:
        conn = http.client.HTTPSConnection("api.gpt.ge")
        conn.request("POST", "/v1/chat/completions", payload, headers)
        res = conn.getresponse()
        data = res.read()
        response_data = json.loads(data.decode("utf-8"))
        ai_reply = response_data.get('choices')[0].get('message').get('content', '')
    except Exception as e:
        return jsonify({'error': f'Error during ChatGPT call: {str(e)}'}), 500

    # Append the assistant's reply to the conversation history.
    conversation.append({
        "role": "assistant",
        "content": ai_reply
    })

    # Generate dynamic suggested follow-up questions using the same format
    suggested_questions = generate_suggested_questions(user_message, ai_reply, conversation)

    # Save/update the conversation in the database.
    conversation_json = json.dumps(conversation)
    if chat_history_record:
        chat_history_record.conversation = conversation_json
    else:
        chat_history_record = ChatHistory(
            user_id=current_user.id,
            conversation=conversation_json
        )
        db.session.add(chat_history_record)
    db.session.commit()

    return jsonify({
        'reply': ai_reply,
        'suggested_questions': suggested_questions  # Include suggested questions in the response
    })

def generate_suggested_questions(user_message, ai_reply, conversation):
    # Create a more context-rich prompt for ChatGPT to generate follow-up questions
    prompt = (
        f"Based on the following user message and AI response, suggest relevant follow-up questions only:\n\n"
        f"User: '{user_message}'\n"
        f"AI: '{ai_reply}'\n"
        f"Please suggest only questions that I, as the user, might ask:"
    )

    # Build the payload for the suggestion request
    payload = json.dumps({
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.5,
        "stream": False
    })

    headers = {
        'Authorization': 'Bearer sk-jqmhNrC6MgA7pxLP80E880095eDa4029AfB1Ef37E15bDa1c',  # Use your API key securely.
        'Content-Type': 'application/json'
    }

    try:
        conn = http.client.HTTPSConnection("api.gpt.ge")
        conn.request("POST", "/v1/chat/completions", payload, headers)
        res = conn.getresponse()
        data = res.read()
        response_data = json.loads(data.decode("utf-8"))
        questions = response_data.get('choices')[0].get('message').get('content', '').strip().split('\n')

        # Limit to a maximum of 3 questions
        formatted_questions = [q.replace("You", "I") for q in questions if q.strip()][:3]  # Adjust phrasing

        return formatted_questions
    except Exception as e:
        return ["Could not generate relevant questions at this time."]



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
@login_required
def test_api():
    user_email = current_user.email
    user_id = current_user.id
    user_name = current_user.name
    user_age = current_user.age
    return render_template('dashboard.html', is_logged_in=True, user_email=user_email, user_id=user_id,user_name=user_name, user_age=user_age)

@app.route('/user_profile', methods=['GET'])
@login_required
def user_profile():
    user_info = {
        'email': current_user.email,
        'name': current_user.name,
        'age': current_user.age,
        'id': current_user.id
    }
    return jsonify(user_info)


@app.route('/api/chat/history', methods=['GET'])
@login_required
def get_chat_history():
    chat_history_record = ChatHistory.query.filter_by(user_id=current_user.id).first()
    if chat_history_record:
        try:
            conversation = json.loads(chat_history_record.conversation)
            return jsonify({'history': conversation})
        except:
            return jsonify({'history': []})
    return jsonify({'history': []}) 


with app.app_context():
    db.create_all()  # Create the database tables if they do not exist

if __name__ == '__main__':
    app.run(debug=True)