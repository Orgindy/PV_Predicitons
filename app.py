# app.py - Streamlit RC + PV Performance Explorer
import streamlit as st
import numpy as np
import pandas as pd
import argparse
import os
from scipy.spatial import cKDTree
import plotly.express as px
import plotly.graph_objects as go
import io
import zipfile
import warnings
warnings.filterwarnings('ignore')


def parse_args():
    parser = argparse.ArgumentParser(description="Run Streamlit PV dashboard")
    parser.add_argument(
        "--data-path",
        default="matched_dataset.csv",
        help="Path to matched dataset CSV",
    )
    parser.add_argument("--db-url", default=os.getenv("PV_DB_URL"))
    parser.add_argument("--db-table", default=os.getenv("PV_DB_TABLE", "pv_data"))
    return parser.parse_args()

ARGS = parse_args()
DATA_PATH = ARGS.data_path
DB_URL = ARGS.db_url
DB_TABLE = ARGS.db_table

def compute_temperature_series(ghi, tair, ir_down, wind, zenith, material_config,
                             switching_profile, emissivity_profile, alpha_profile):
    """
    Simplified thermal modeling for RC surface temperature.
    
    Parameters:
    - ghi: Global Horizontal Irradiance (W/m²)
    - tair: Air temperature (K)
    - ir_down: Downward longwave radiation from the sky (W/m²). This is used to
      estimate the effective sky temperature.
    - wind: Wind speed (m/s)
    - zenith: Solar zenith angle (degrees)
    - material_config: Dict with material properties
    - switching_profile: Dict with switching states
    - emissivity_profile: Dict with emissivity values
    - alpha_profile: Dict with absorptivity values
    
    Returns:
    - T_rc: RC surface temperature array (K)
    """
    ghi = np.array(ghi)
    tair = np.array(tair)
    ir_down = np.array(ir_down)
    wind = np.array(wind)
    zenith = np.array(zenith)
    
    # Extract material properties
    epsilon = material_config.get('epsilon_IR', 0.92)
    alpha = material_config.get('alpha_solar', 0.85)
    h_conv_base = material_config.get('h_conv_base', 5)
    h_conv_wind = material_config.get('h_conv_wind_coeff', 4)
    
    # Convective heat transfer coefficient
    h_conv = h_conv_base + h_conv_wind * wind
    
    # Solar heating (considering zenith angle)
    zenith_rad = np.radians(zenith)
    cos_zenith = np.cos(zenith_rad)
    cos_zenith = np.clip(cos_zenith, 0, 1)  # Only positive values
    
    Q_solar = alpha * ghi * cos_zenith
    
    # Radiative cooling to sky (simplified)
    sigma = 5.67e-8  # Stefan-Boltzmann constant
    # Estimate effective sky temperature from downwelling IR
    T_sky = (ir_down / (epsilon * sigma)) ** 0.25
    
    # Iterative solution for RC temperature
    T_rc = tair.copy()  # Initial guess
    
    for _ in range(5):  # Simple iteration
        Q_rad_cooling = epsilon * sigma * (T_rc**4 - T_sky**4)
        Q_conv = h_conv * (T_rc - tair)
        
        # Energy balance: Q_solar = Q_rad_cooling + Q_conv
        # Simplified: T_rc = T_air + (Q_solar - Q_rad_cooling) / h_conv
        dT = (Q_solar - Q_rad_cooling) / (h_conv + 4 * epsilon * sigma * T_rc**3)
        T_rc = tair + dT
        
        # Ensure reasonable temperature range
        T_rc = np.clip(T_rc, tair - 20, tair + 30)
    
    return T_rc

def estimate_pv_cell_temperature(ghi, tair_c, wind, model="NOCT"):
    """
    Estimate PV cell temperature using NOCT model.
    
    Parameters:
    - ghi: Global Horizontal Irradiance (W/m²)
    - tair_c: Air temperature (°C)
    - wind: Wind speed (m/s)
    - model: Temperature model ("NOCT", "Sandia", etc.)
    
    Returns:
    - T_pv: PV cell temperature (°C)
    """
    ghi = np.array(ghi)
    tair_c = np.array(tair_c)
    wind = np.array(wind)
    
    if model == "NOCT":
        # NOCT (Nominal Operating Cell Temperature) model
        NOCT = 45  # °C (typical for crystalline silicon)
        G_ref = 800  # W/m² (reference irradiance)
        T_ref = 20  # °C (reference temperature)
        
        # Basic NOCT formula
        T_pv = tair_c + (NOCT - T_ref) * (ghi / G_ref)
        
        # Wind cooling effect (simplified)
        wind_cooling = np.clip(wind - 1, 0, 10) * 0.5  # 0.5°C cooling per m/s above 1 m/s
        T_pv = T_pv - wind_cooling
        
    elif model == "Sandia":
        # Simplified Sandia model
        a = -3.56  # Sandia parameter
        b = -0.075  # Sandia parameter
        deltaT = 3  # Temperature difference factor
        
        T_pv = ghi * np.exp(a + b * wind) + tair_c + (ghi / 1000) * deltaT
        
    else:
        # Fallback to simple linear model
        T_pv = tair_c + (ghi / 800) * 25
    
    return T_pv

def calculate_synergy_index(T_pv, T_rc, ghi, gamma_pv=-0.004):
    """
    Calculate synergy index between RC cooling and PV performance.
    
    Parameters:
    - T_pv: PV cell temperature (°C)
    - T_rc: RC surface temperature (°C)
    - ghi: Global Horizontal Irradiance (W/m²)
    - gamma_pv: PV temperature coefficient (%/°C)
    
    Returns:
    - synergy_index: Synergy index (%)
    """
    T_pv = np.array(T_pv)
    T_rc = np.array(T_rc)
    ghi = np.array(ghi)
    
    # Calculate temperature reduction from RC cooling
    temp_reduction = np.mean(T_pv) - np.mean(T_rc)
    
    # Calculate PV power gain from temperature reduction
    power_gain = temp_reduction * abs(gamma_pv) * 100  # Convert to percentage
    
    # Weight by irradiance (more benefit during high solar conditions)
    irradiance_weight = np.mean(ghi) / 1000  # Normalize to 1000 W/m²
    
    synergy_index = power_gain * irradiance_weight
    
    return synergy_index

st.set_page_config(
    page_title="RC + PV Performance Tool", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #d4d4d4;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🌞 Radiative Cooling + PV Performance Explorer")
st.markdown("""
**Purpose:** Analyze the synergy between radiative cooling (RC) and photovoltaic (PV) systems.
Enter your location and material properties to simulate performance and find optimal zones.
""")

# Information expander
with st.expander("ℹ️ How this tool works"):
    st.markdown("""
    1. **Location Matching**: Find the closest climate zone in our dataset
    2. **Material Simulation**: Model RC surface temperature and PV cell temperature
    3. **Synergy Calculation**: Quantify the performance benefit of RC cooling on PV
    4. **Zone Recommendations**: Identify similar high-performance locations
    5. **Visualization**: Interactive maps and downloadable results
    """)
    
@st.cache_data
def load_cluster_dataset(csv_path=DATA_PATH, db_url=DB_URL, db_table=DB_TABLE):
    """
    Load cluster dataset with comprehensive error handling and column detection.
    
    Returns:
    - df: DataFrame with location and cluster data
    - tree: cKDTree for fast spatial queries
    """
    try:
        if db_url:
            from database_utils import read_table
            df = read_table(db_table, db_url=db_url)
        else:
            df = pd.read_csv(csv_path)
        st.success(f"✅ Dataset loaded: {len(df)} locations")
        
        # Flexible column detection
        coordinate_mapping = {}
        
        # Find latitude column
        lat_options = ['lat', 'latitude', 'LAT', 'Latitude']
        for col in lat_options:
            if col in df.columns:
                coordinate_mapping['lat'] = col
                break
        
        # Find longitude column
        lon_options = ['lon', 'longitude', 'LON', 'Longitude', 'lng']
        for col in lon_options:
            if col in df.columns:
                coordinate_mapping['lon'] = col
                break
        
        if len(coordinate_mapping) < 2:
            st.error("❌ Could not find latitude/longitude columns in dataset")
            st.write("Available columns:", df.columns.tolist())
            return None, None
        
        # Standardize column names
        df = df.rename(columns={
            coordinate_mapping['lat']: 'lat',
            coordinate_mapping['lon']: 'lon'
        })
        
        # Validate coordinate ranges
        lat_valid = df['lat'].between(-90, 90).all()
        lon_valid = df['lon'].between(-180, 180).all()
        
        if not lat_valid or not lon_valid:
            st.warning("⚠️ Some coordinates are outside valid ranges")
        
        # Create spatial index
        coords = df[['lat', 'lon']].values
        tree = cKDTree(coords)
        
        # Display dataset info
        with st.expander("📊 Dataset Information"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Locations", len(df))
            with col2:
                if 'Cluster_ID' in df.columns:
                    st.metric("Clusters", df['Cluster_ID'].nunique())
                else:
                    st.metric("Clusters", "N/A")
            with col3:
                if 'Best_Technology' in df.columns:
                    st.metric("Technologies", df['Best_Technology'].nunique())
                else:
                    st.metric("Technologies", "N/A")
            
            st.write("**Available columns:**", df.columns.tolist())
        
        return df, tree
        
    except FileNotFoundError:
        st.error(f"❌ Dataset file not found: {csv_path}")
        st.info(
            f"""
        **To use this tool:**
        1. Run `main.py` first to generate the matched dataset
        2. Ensure the file `{csv_path}` exists in your project directory
        3. Refresh this page
        """
        )
        return None, None
        
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        return None, None

def match_location(lat, lon, df, tree, k=1):
    """
    Find closest location(s) in dataset to user coordinates.
    
    Parameters:
    - lat, lon: User coordinates
    - df: Dataset DataFrame
    - tree: cKDTree for spatial queries
    - k: Number of nearest neighbors to return
    
    Returns:
    - match: Closest location data
    - distance: Distance to closest location (degrees)
    - neighbors: k nearest neighbors (if k > 1)
    """
    distances, indices = tree.query([lat, lon], k=k)
    
    if k == 1:
        return df.iloc[indices], distances
    else:
        return df.iloc[indices], distances
    
    # Sidebar for inputs
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Location Input
    st.subheader("📍 Location")
    lat = st.number_input(
        "Latitude", 
        min_value=-90.0, 
        max_value=90.0, 
        value=52.0, 
        step=0.1,
        help="Enter latitude in decimal degrees (-90 to 90)"
    )
    lon = st.number_input(
        "Longitude", 
        min_value=-180.0, 
        max_value=180.0, 
        value=5.0, 
        step=0.1,
        help="Enter longitude in decimal degrees (-180 to 180)"
    )
    
    # Material Properties
    st.subheader("🔬 Material Properties")
    
    # Preset material options
    material_preset = st.selectbox(
        "Material Preset",
        ["Custom", "Standard RC", "High-Performance RC", "Selective Emitter"],
        help="Choose a preset or select 'Custom' for manual input"
    )

    # Default material properties to avoid undefined variables
    epsilon = 0.92
    alpha = 0.85

    if material_preset == "Custom":
        epsilon = st.slider(
            "Emissivity (ε)",
            0.70, 1.00, 0.92, 0.01,
            help="IR emissivity for radiative cooling"
        )
        alpha = st.slider(
            "Solar Absorptivity (α)", 
            0.00, 1.00, 0.85, 0.01,
            help="Solar absorptivity of RC surface"
        )
    elif material_preset == "Standard RC":
        epsilon, alpha = 0.92, 0.85
    elif material_preset == "High-Performance RC":
        epsilon, alpha = 0.95, 0.05
    elif material_preset == "Selective Emitter":
        epsilon, alpha = 0.85, 0.15
    
    # PV Properties
    st.subheader("⚡ PV Properties")
    gamma = st.slider(
        "Temperature Coefficient (γ, %/°C)", 
        -0.01, 0.0, -0.004, 0.0005,
        help="PV power temperature coefficient (typically negative)"
    )
    
    pv_model = st.selectbox(
        "PV Temperature Model",
        ["NOCT", "Sandia", "Simple"],
        help="Model for estimating PV cell temperature"
    )
    
    # Simulation Settings
    st.subheader("⚙️ Simulation Settings")
    
    num_neighbors = st.slider(
        "Number of Similar Zones", 
        1, 10, 5,
        help="Number of similar zones to recommend"
    )
    
    show_advanced = st.checkbox("Show Advanced Metrics", value=False)
    
# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎯 Simulation Results")
    
    # Load dataset
    data_loading_status = st.empty()
    with data_loading_status:
        with st.spinner("Loading dataset..."):
            df, tree = load_cluster_dataset(db_url=DB_URL, db_table=DB_TABLE)
    
    if df is not None and tree is not None:
        data_loading_status.empty()
        
        # Run simulation button
        if st.button("🚀 Run Performance Simulation", type="primary"):
            with st.spinner("Running simulation..."):
                
                # Display input summary
                st.write("### 📋 Input Summary")
                input_col1, input_col2 = st.columns(2)
                
                with input_col1:
                    st.write(f"**Location:** ({lat:.3f}°, {lon:.3f}°)")
                    st.write(f"**Material:** {material_preset}")
                    st.write(f"**Emissivity:** {epsilon:.2f}")
                
                with input_col2:
                    st.write(f"**Absorptivity:** {alpha:.2f}")
                    st.write(f"**PV Coefficient:** {gamma:.4f} %/°C")
                    st.write(f"**PV Model:** {pv_model}")
                
                # Find matching location
                match, distance = match_location(lat, lon, df, tree)
                
                # Display matched zone info
                st.write("### 🎯 Matched Climate Zone")
                
                match_col1, match_col2, match_col3 = st.columns(3)
                
                with match_col1:
                    st.metric(
                        "Distance to Match", 
                        f"{distance:.2f}°",
                        help="Distance to nearest data point"
                    )
                    st.metric(
                        "Cluster ID", 
                        match.get('Cluster_ID', 'N/A')
                    )
                
                with match_col2:
                    ghi_val = match.get('GHI', match.get('ghi', 0))
                    st.metric("Solar Irradiance", f"{ghi_val:.0f} W/m²")
                    
                    rc_val = match.get('RC_Potential', match.get('RC_potential', match.get('rc_potential', 0)))
                    st.metric("RC Potential", f"{rc_val:.0f} W/m²")
                
                with match_col3:
                    if 'Best_Technology' in match:
                        st.metric("Best PV Tech", match['Best_Technology'])
                    
                    if 'Synergy_Index' in match:
                        st.metric("Historical Synergy", f"{match['Synergy_Index']:.1f}%")
                
                # Create hourly simulation data
                st.write("### 🔬 Hourly Simulation")
                
                # Generate realistic hourly profiles
                hours = np.arange(24)
                
                # Solar irradiance profile (bell curve)
                ghi_profile = ghi_val * np.maximum(0, np.sin(np.pi * (hours - 6) / 12))
                ghi_profile = np.clip(ghi_profile, 0, ghi_val)
                
                # Temperature profile (sinusoidal)
                temp_base = match.get('T_air', match.get('Temperature', 15))
                if temp_base > 100:  # Convert from Kelvin if needed
                    temp_base = temp_base - 273.15
                
                temp_profile_c = temp_base + 5 * np.sin(np.pi * (hours - 6) / 12)
                temp_profile_k = temp_profile_c + 273.15
                
                # Wind profile (variable)
                wind_profile = 2 + np.random.normal(0, 0.5, 24)
                wind_profile = np.clip(wind_profile, 0.5, 10)
                
                # IR radiation profile
                ir_profile = np.ones(24) * 350  # Simplified constant
                
                # Zenith angle profile
                zenith_profile = np.concatenate([
                    np.linspace(90, 0, 12),  # Morning
                    np.linspace(0, 90, 12)   # Afternoon
                ])
                
                # Material configuration
                material_config = {
                    "alpha_solar": alpha,
                    "epsilon_IR": epsilon,
                    "thickness_m": 0.003,
                    "density": 2500,
                    "cp": 900,
                    "h_conv_base": 5,
                    "h_conv_wind_coeff": 4,
                    "use_dynamic_emissivity": False
                }
                
                # Run thermal simulations
                T_rc_k = compute_temperature_series(
                    ghi_profile, temp_profile_k, ir_profile, wind_profile, zenith_profile,
                    material_config,
                    switching_profile={"state_map": {}, "default": "static"},
                    emissivity_profile={"default": epsilon},
                    alpha_profile={"default": alpha}
                )
                
                T_pv_c = estimate_pv_cell_temperature(ghi_profile, temp_profile_c, wind_profile, model=pv_model)
                T_rc_c = T_rc_k - 273.15
                
                # Calculate synergy index
                synergy_index = calculate_synergy_index(T_pv_c, T_rc_c, ghi_profile, gamma_pv=gamma)
                
                # Display simulation results
                st.write("### 📊 Performance Metrics")
                
                perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
                
                with perf_col1:
                    st.metric(
                        "Avg PV Temperature", 
                        f"{np.mean(T_pv_c):.1f}°C",
                        help="Average PV cell temperature"
                    )
                
                with perf_col2:
                    st.metric(
                        "Avg RC Temperature", 
                        f"{np.mean(T_rc_c):.1f}°C",
                        help="Average RC surface temperature"
                    )
                
                with perf_col3:
                    temp_reduction = np.mean(T_pv_c) - np.mean(T_rc_c)
                    st.metric(
                        "Temperature Reduction", 
                        f"{temp_reduction:.1f}°C",
                        help="RC cooling benefit"
                    )
                
                with perf_col4:
                    st.metric(
                        "Synergy Index", 
                        f"{synergy_index:.2f}%",
                        help="Overall performance benefit"
                    )
                    
                    
# Hourly temperature chart
                st.write("### 📈 Hourly Temperature Profiles")
                
                # Create temperature comparison chart
                fig_temp = go.Figure()
                
                fig_temp.add_trace(go.Scatter(
                    x=hours,
                    y=temp_profile_c,
                    mode='lines',
                    name='Air Temperature',
                    line=dict(color='blue', width=2)
                ))
                
                fig_temp.add_trace(go.Scatter(
                    x=hours,
                    y=T_pv_c,
                    mode='lines',
                    name='PV Cell Temperature',
                    line=dict(color='red', width=2)
                ))
                
                fig_temp.add_trace(go.Scatter(
                    x=hours,
                    y=T_rc_c,
                    mode='lines',
                    name='RC Surface Temperature',
                    line=dict(color='green', width=2)
                ))
                
                fig_temp.update_layout(
                    title="Temperature Profiles Throughout the Day",
                    xaxis_title="Hour of Day",
                    yaxis_title="Temperature (°C)",
                    legend=dict(x=0.02, y=0.98),
                    height=400,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig_temp, use_container_width=True)
                
                # Solar irradiance chart
                fig_solar = go.Figure()
                
                fig_solar.add_trace(go.Scatter(
                    x=hours,
                    y=ghi_profile,
                    mode='lines+markers',
                    name='Solar Irradiance',
                    line=dict(color='orange', width=3),
                    fill='tonexty'
                ))
                
                fig_solar.update_layout(
                    title="Solar Irradiance Profile",
                    xaxis_title="Hour of Day",
                    yaxis_title="Irradiance (W/m²)",
                    height=300,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig_solar, use_container_width=True)
                
                # Advanced metrics (if enabled)
                if show_advanced:
                    st.write("### 🔬 Advanced Analysis")
                    
                    adv_col1, adv_col2 = st.columns(2)
                    
                    with adv_col1:
                        st.write("**Thermal Analysis:**")
                        st.write(f"• Max PV Temperature: {np.max(T_pv_c):.1f}°C")
                        st.write(f"• Max RC Temperature: {np.max(T_rc_c):.1f}°C")
                        st.write(f"• Temperature Variance: {np.var(T_pv_c - T_rc_c):.2f}")
                        
                    with adv_col2:
                        st.write("**Energy Analysis:**")
                        daily_energy = np.trapz(ghi_profile) / 1000  # kWh/m²/day
                        st.write(f"• Daily Solar Energy: {daily_energy:.2f} kWh/m²")
                        st.write(f"• Peak Sun Hours: {daily_energy:.1f} hours")
                        cooling_hours = np.sum(T_rc_c < T_pv_c)
                        st.write(f"• Cooling Hours: {cooling_hours}/24")
                        
# Find similar zones
                st.write("### 🌍 Recommended Similar Zones")
                
                cluster_id = match.get('Cluster_ID', 0)
                if cluster_id != 0 and 'Cluster_ID' in df.columns:
                    # Get locations in same cluster
                    similar_zones = df[df['Cluster_ID'] == cluster_id].copy()
                    
                    if len(similar_zones) > 1:
                        # Calculate distance from user location to all similar zones
                        similar_coords = similar_zones[['lat', 'lon']].values
                        user_coord = np.array([[lat, lon]])
                        distances = np.linalg.norm(similar_coords - user_coord, axis=1)
                        similar_zones['distance_to_user'] = distances
                        
                        # Sort by relevant metric
                        sort_options = ['Synergy_Index', 'RC_Potential', 'RC_potential', 'GHI', 'distance_to_user']
                        sort_col = None
                        for col in sort_options:
                            if col in similar_zones.columns:
                                sort_col = col
                                break
                        
                        if sort_col == 'distance_to_user':
                            similar_zones = similar_zones.sort_values(sort_col, ascending=True)
                        else:
                            similar_zones = similar_zones.sort_values(sort_col, ascending=False)
                        
                        # Select top zones
                        top_zones = similar_zones.head(num_neighbors)
                        
                        # Display recommendations
                        for idx, (_, zone) in enumerate(top_zones.iterrows()):
                            col_a, col_b, col_c, col_d = st.columns(4)
                            
                            with col_a:
                                st.write(f"**Zone {idx+1}**")
                                st.write(f"({zone['lat']:.2f}, {zone['lon']:.2f})")
                            
                            with col_b:
                                synergy_val = zone.get('Synergy_Index', 'N/A')
                                st.write(f"Synergy: {synergy_val}")
                            
                            with col_c:
                                rc_val = zone.get('RC_Potential', zone.get('RC_potential', 0))
                                st.write(f"RC: {rc_val:.0f} W/m²")
                            
                            with col_d:
                                ghi_val = zone.get('GHI', zone.get('ghi', 0))
                                st.write(f"GHI: {ghi_val:.0f} W/m²")
                        
                        # Create map of recommended zones
                        st.write("### 🗺️ Recommended Zones Map")
                        
                        # Prepare map data
                        map_data = top_zones[['lat', 'lon']].copy()
                        map_data['Type'] = 'Recommended Zone'
                        map_data['Size'] = 8
                        
                        # Add user location
                        user_data = pd.DataFrame({
                            'lat': [lat],
                            'lon': [lon],
                            'Type': ['Your Location'],
                            'Size': [12]
                        })
                        
                        map_data = pd.concat([map_data, user_data], ignore_index=True)
                        
                        # Create interactive map
                        fig_map = px.scatter_geo(
                            map_data,
                            lat='lat',
                            lon='lon',
                            color='Type',
                            size='Size',
                            title="Your Location and Recommended Zones",
                            projection="natural earth",
                            template="plotly_white",
                            color_discrete_map={
                                'Your Location': 'red',
                                'Recommended Zone': 'blue'
                            }
                        )
                        
                        fig_map.update_traces(
                            marker=dict(
                                line=dict(width=1, color='black'),
                                sizemode='diameter'
                            )
                        )
                        
                        fig_map.update_layout(
                            height=500,
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_map, use_container_width=True)
                    
                    else:
                        st.info("No similar zones found in the same cluster.")
                else:
                    st.info("Cluster information not available for zone recommendations.")
                    
# Results export section
                st.write("### 💾 Export Results")
                
                # Prepare results data
                results_data = {
                    'Input_Latitude': lat,
                    'Input_Longitude': lon,
                    'Material_Preset': material_preset,
                    'Emissivity': epsilon,
                    'Absorptivity': alpha,
                    'PV_Temperature_Coefficient': gamma,
                    'PV_Model': pv_model,
                    'Matched_Cluster': cluster_id,
                    'Distance_to_Match_deg': distance,
                    'Avg_PV_Temperature_C': np.mean(T_pv_c),
                    'Avg_RC_Temperature_C': np.mean(T_rc_c),
                    'Temperature_Reduction_C': np.mean(T_pv_c) - np.mean(T_rc_c),
                    'Synergy_Index_Percent': synergy_index,
                    'Daily_Solar_Energy_kWh_m2': np.trapz(ghi_profile) / 1000,
                    'Cooling_Hours_per_Day': np.sum(T_rc_c < T_pv_c)
                }
                
                # Convert to DataFrame
                results_df = pd.DataFrame([results_data])
                
                # Hourly data
                hourly_data = pd.DataFrame({
                    'Hour': hours,
                    'Air_Temperature_C': temp_profile_c,
                    'PV_Temperature_C': T_pv_c,
                    'RC_Temperature_C': T_rc_c,
                    'Solar_Irradiance_W_m2': ghi_profile,
                    'Wind_Speed_m_s': wind_profile,
                    'Temperature_Difference_C': T_pv_c - T_rc_c
                })
                
                # Download options
                export_col1, export_col2, export_col3 = st.columns(3)
                
                with export_col1:
                    # CSV download for summary results
                    csv_summary = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📊 Download Summary CSV",
                        data=csv_summary,
                        file_name=f"rc_pv_summary_{lat}_{lon}.csv",
                        mime="text/csv",
                        help="Download simulation summary results"
                    )
                
                with export_col2:
                    # CSV download for hourly data
                    csv_hourly = hourly_data.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⏰ Download Hourly CSV",
                        data=csv_hourly,
                        file_name=f"rc_pv_hourly_{lat}_{lon}.csv",
                        mime="text/csv",
                        help="Download hourly simulation data"
                    )
                
                with export_col3:
                    # Create comprehensive ZIP bundle
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        # Add summary CSV
                        zipf.writestr("summary_results.csv", csv_summary)
                        
                        # Add hourly CSV
                        zipf.writestr("hourly_data.csv", csv_hourly.encode('utf-8'))
                        
                        # Add configuration file
                        config_text = f"""RC-PV Simulation Configuration
Location: {lat:.3f}°, {lon:.3f}°
Material Preset: {material_preset}
Emissivity: {epsilon:.2f}
Absorptivity: {alpha:.2f}
PV Temperature Coefficient: {gamma:.4f} %/°C
PV Model: {pv_model}
Simulation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                        zipf.writestr("configuration.txt", config_text)
                        
                        # Add map as PNG (if possible)
                        try:
                            map_img = fig_map.to_image(format="png", engine="kaleido")
                            zipf.writestr("recommended_zones_map.png", map_img)
                        except Exception as e:
                            pass  # Skip if image export fails
                        
                        # Add temperature chart as PNG
                        try:
                            temp_img = fig_temp.to_image(format="png", engine="kaleido")
                            zipf.writestr("temperature_profiles.png", temp_img)
                        except Exception as e:
                            pass  # Skip if image export fails
                    
                    zip_buffer.seek(0)
                    
                    st.download_button(
                        label="📦 Download Complete Report",
                        data=zip_buffer,
                        file_name=f"rc_pv_report_{lat}_{lon}.zip",
                        mime="application/zip",
                        help="Download complete analysis bundle (CSV + images + config)"
                    )
                
                # Success message
                st.success("✅ Simulation completed successfully!")
                
        else:
            st.info("👆 Click the button above to run the performance simulation")
    
    else:
        st.error("❌ Cannot load dataset. Please ensure the required files are available.")       
    
        with col2:
            st.subheader("📋 Quick Info")
    
    # Dataset status
    if df is not None:
        st.success("✅ Dataset loaded successfully")
        
        with st.expander("📊 Dataset Statistics"):
            st.write(f"**Total locations:** {len(df):,}")
            
            if 'Cluster_ID' in df.columns:
                n_clusters = df['Cluster_ID'].nunique()
                st.write(f"**Climate clusters:** {n_clusters}")
            
            if 'Best_Technology' in df.columns:
                tech_counts = df['Best_Technology'].value_counts()
                st.write("**Technology distribution:**")
                for tech, count in tech_counts.head(5).items():
                    st.write(f"• {tech}: {count:,} locations")
            
            # Geographic coverage
            if 'lat' in df.columns and 'lon' in df.columns:
                lat_range = df['lat'].max() - df['lat'].min()
                lon_range = df['lon'].max() - df['lon'].min()
                st.write(f"**Geographic coverage:**")
                st.write(f"• Latitude: {df['lat'].min():.1f}° to {df['lat'].max():.1f}°")
                st.write(f"• Longitude: {df['lon'].min():.1f}° to {df['lon'].max():.1f}°")
    
    else:
        st.error("❌ Dataset not available")
    
    # Help section
    st.subheader("❓ Help & Tips")
    
    with st.expander("🎯 How to interpret results"):
        st.markdown("""
        **Synergy Index:** Percentage improvement in PV performance due to RC cooling
        - **Positive values:** RC cooling benefits PV performance
        - **Higher values:** Better synergy between RC and PV
        
        **Temperature Reduction:** Difference between PV and RC surface temperatures
        - **Positive values:** RC surface is cooler than PV
        - **Higher values:** More effective cooling
        """)
    
    with st.expander("🔧 Material selection guide"):
        st.markdown("""
        **High Emissivity (ε > 0.9):** Better radiative cooling
        **Low Absorptivity (α < 0.2):** Less solar heating
        **Selective materials:** High ε, low α combination
        
        **Preset recommendations:**
        - **Standard RC:** Balanced performance
        - **High-Performance:** Maximum cooling
        - **Selective Emitter:** Optimized spectral properties
        """)
    
    with st.expander("⚡ PV temperature models"):
        st.markdown("""
        **NOCT Model:** Simple, widely used
        **Sandia Model:** More accurate, considers wind effects
        **Simple Model:** Basic linear approximation
        
        **Temperature coefficient typically:**
        - Silicon: -0.4 to -0.5 %/°C
        - Thin-film: -0.2 to -0.3 %/°C
        """)
    
    # Contact/About section
    st.subheader("ℹ️ About")
    st.markdown("""
    **RC-PV Performance Explorer**
    
    This tool analyzes the synergy between radiative cooling and photovoltaic systems using:
    - Climate zone clustering
    - Thermal modeling
    - Performance optimization
    
    **Version:** 2.0  
    **Last updated:** 2025
    """)
    
    # Debug information (for development)
    if st.checkbox("🐛 Show debug info", value=False):
        st.subheader("🔍 Debug Information")
        
        if df is not None:
            st.write("**DataFrame shape:**", df.shape)
            st.write("**Column data types:**")
            st.write(df.dtypes)
            
            st.write("**Sample data:**")
            st.dataframe(df.head(3))
        
        st.write("**Session state:**")
        st.write(dict(st.session_state))
        
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🌞 RC-PV Performance Explorer | Built with Streamlit</p>
    <p>Explore the synergy between radiative cooling and photovoltaic systems</p>
</div>
""", unsafe_allow_html=True)

# Additional CSS for mobile responsiveness
st.markdown("""
<style>
    @media (max-width: 768px) {
        .stColumns > div {
            width: 100% !important;
            margin-bottom: 1rem;
        }
        
        .main > div {
            padding: 1rem;
        }
    }
    
    .stMetric > div {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 0.5rem;
    }
    
    .stExpander > details > summary {
        font-weight: 600;
        color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Global error handler
def handle_simulation_error(error, context="simulation"):
    """Global error handler for simulation errors."""
    error_msg = str(error)
    
    if "FileNotFoundError" in error_msg:
        st.error("❌ Required data files are missing")
        st.info("Please run `main.py` first to generate the necessary datasets")
    
    elif "KeyError" in error_msg:
        st.error("❌ Data format error")
        st.info("The dataset may be missing required columns")
    
    elif "ValueError" in error_msg:
        st.error("❌ Invalid input values")
        st.info("Please check your input parameters and try again")
    
    else:
        st.error(f"❌ {context.title()} error: {error_msg}")
        st.info("Please check the logs or contact support if the problem persists")
    
    # Log error for debugging
    import logging
    logging.error(f"{context} error: {error_msg}")

# Input validation functions
def validate_coordinates(lat, lon):
    """Validate latitude and longitude inputs."""
    if not (-90 <= lat <= 90):
        st.error("❌ Latitude must be between -90° and 90°")
        return False
    
    if not (-180 <= lon <= 180):
        st.error("❌ Longitude must be between -180° and 180°")
        return False
    
    return True

def validate_material_properties(epsilon, alpha):
    """Validate material property inputs."""
    if not (0 <= epsilon <= 1):
        st.error("❌ Emissivity must be between 0 and 1")
        return False
    
    if not (0 <= alpha <= 1):
        st.error("❌ Absorptivity must be between 0 and 1")
        return False
    
    return True

# Performance monitoring
def monitor_performance():
    """Monitor app performance and display metrics."""
    if 'start_time' not in st.session_state:
        st.session_state.start_time = pd.Timestamp.now()
    
    current_time = pd.Timestamp.now()
    runtime = (current_time - st.session_state.start_time).total_seconds()
    
    if st.checkbox("⚡ Show performance metrics", value=False):
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            st.metric("App Runtime", f"{runtime:.1f}s")
        
        with perf_col2:
            if df is not None:
                st.metric("Data Loading", "✅ Complete")
            else:
                st.metric("Data Loading", "❌ Failed")

# Call performance monitoring
monitor_performance()
