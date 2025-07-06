import pandas as pd
import firebase_admin
from firebase_admin import credentials, db
import matplotlib.pyplot as plt
import seaborn as sns
import os
import base64
import tempfile

# --- Load Firebase credentials securely from base64 env variable ---
firebase_key_b64 = os.getenv("FIREBASE_KEY_B64")
if not firebase_key_b64:
    # Fallback for local development if environment variable is not set
    try:
        cred = credentials.Certificate("agriastrax-website-firebase-adminsdk-fbsvc-36cdff39c2.json")
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred, {
                'databaseURL': 'https://agriastrax-website-default-rtdb.firebaseio.com/'
            })
        print("Firebase initialized using local credentials file.")
    except Exception as e:
        raise ValueError(f"❌ FIREBASE_KEY_B64 environment variable not set and local credentials file not found or invalid: {e}")
else:
    decoded = base64.b64decode(firebase_key_b64).decode("utf-8")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as f:
        f.write(decoded)
        key_path = f.name

    # --- Firebase Setup ---
    if not firebase_admin._apps:
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred, {
            'databaseURL': 'https://agriastrax-website-default-rtdb.firebaseio.com/'
        })
    # Clean up the temporary file after initialization for security
    os.remove(key_path)
    print("Firebase initialized using environment variable.")


# --- Fetch Data ---
def fetch_data():
    """Fetches all sensor data from Firebase Realtime Database."""
    ref = db.reference('sensors/farm1')
    snapshot = ref.get()
    if not snapshot:
        print("❌ No data found in Firebase.")
        return pd.DataFrame()
    
    # Convert Firebase snapshot to DataFrame, transpose to get columns as features
    df = pd.DataFrame(snapshot).T
    return df

# --- Preprocessing ---
def preprocess(df):
    """
    Cleans and prepares the DataFrame for EDA.
    Converts timestamp, sorts, and attempts to convert all columns to numeric,
    handling errors gracefully.
    """
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp']) # Drop rows where timestamp conversion failed
        df = df.sort_values('timestamp')
    else:
        print("Warning: 'timestamp' column not found.")

    # Attempt to convert all relevant columns to numeric, ignoring errors
    # This is crucial for plotting and analysis
    numeric_cols = ['soil_moisture', 'temperature', 'humidity', 'pH', 'light_intensity', 
                    'N', 'P', 'K', 'rainfall', 'growth_factor', 'growth_trigger']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = df.reset_index(drop=True)
    return df

# --- EDA Plotting ---
def plot_eda(df):
    """
    Generates time-series plots for sensor data, growth factors, and crop stages.
    Uses subplots for better clarity.
    """
    sns.set(style="darkgrid")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 18), sharex=True)
    fig.suptitle('Smart AgriTech Sensor Data Analysis', fontsize=16)

    # Plot 1: Core IoT Sensor Time Series
    ax1 = axes[0]
    core_sensor_cols = ['soil_moisture', 'temperature', 'humidity', 'pH', 'light_intensity']
    for col in core_sensor_cols:
        if col in df.columns and not df[col].isnull().all():
            ax1.plot(df['timestamp'], df[col], label=col, marker='.', markersize=4, linestyle='-')
    ax1.legend(title='Sensor Metric', loc='upper left', bbox_to_anchor=(1, 1))
    ax1.set_title('Core IoT Sensor Readings Over Time')
    ax1.set_ylabel('Value')
    ax1.grid(True)

    # Plot 2: Biological Behavior - Growth Factor and Growth Trigger
    ax2 = axes[1]
    if 'growth_factor' in df.columns and not df['growth_factor'].isnull().all():
        sns.lineplot(data=df, x='timestamp', y='growth_factor', label='Growth Factor', ax=ax2, color='green', marker='o', markersize=4)
    else:
        print("Warning: 'growth_factor' column not found or is empty. Skipping plot.")

    if 'growth_trigger' in df.columns and not df['growth_trigger'].isnull().all():
        # Plot growth_trigger as steps for binary changes
        sns.lineplot(data=df, x='timestamp', y='growth_trigger', label='Growth Trigger (Binary)', ax=ax2, color='purple', drawstyle='steps-post', marker='x', markersize=6)
        ax2.set_yticks([0, 1]) # Ensure y-axis shows 0 and 1 clearly
        ax2.set_yticklabels(['No Growth', 'Growth Triggered'])
    else:
        print("Warning: 'growth_trigger' column not found or is empty. Skipping plot.")
        
    ax2.legend(title='Biological Metric', loc='upper left', bbox_to_anchor=(1, 1))
    ax2.set_title('Simulated Biological Behaviors: Growth Factor & Trigger')
    ax2.set_ylabel('Value / State')
    ax2.grid(True)

    # Plot 3: Categorical Biological Behavior - Crop Stage
    ax3 = axes[2]
    if 'crop_stage' in df.columns and not df['crop_stage'].isnull().all():
        # Use scatter plot for categorical data over time
        sns.scatterplot(data=df, x='timestamp', y='crop_stage', ax=ax3, hue='crop_stage', s=100, alpha=0.7, legend='full')
        ax3.set_title('Simulated Crop Stage Over Time')
        ax3.set_ylabel('Crop Stage')
        ax3.legend(title='Crop Stage', loc='upper left', bbox_to_anchor=(1, 1))
    else:
        print("Warning: 'crop_stage' column not found or is empty. Skipping plot.")
        
    ax3.set_xlabel('Timestamp')
    ax3.grid(True)

    # Rotate x-axis labels for better readability
    for ax in axes:
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout to prevent title overlap
    plt.show()

# --- Run the Script ---
if __name__ == '__main__':
    df_raw = fetch_data()
    if not df_raw.empty:
        df_clean = preprocess(df_raw)
        print("\n--- Cleaned Data Head ---")
        print(df_clean.head())
        
        print("\n--- Data Info ---")
        df_clean.info()

        print("\n--- Plotting EDA ---")
        plot_eda(df_clean)
        print("EDA plots displayed.")
    else:
        print("No sensor data available to perform EDA.")