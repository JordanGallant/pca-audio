import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.preprocessing import MinMaxScaler
import matplotlib.patheffects as PathEffects
import random
from sklearn.cluster import KMeans

# Get data
try:
    df = pd.read_json('songs.json')
    print("Successfully loaded songs.json")
    print(f"Dataset shape: {df.shape}")
    print(df.head())
except FileNotFoundError:
    print("Error: 'songs.json' not found. Please ensure the file exists in the current directory.")
    df = None
except Exception as e:
    print(f"An unexpected error occurred: {e}")
    df = None

# Continue only if data was loaded successfully
if df is not None:
    # Clean and normalize the data
    # Check energy column format - it seems to be a large number instead of 0-1 range
    if 'energy' in df.columns:
        # Check if energy has unusually large values
        if df['energy'].mean() > 10:  # If energy is not in 0-1 range
            print("Normalizing energy values...")
            # Normalize energy to 0-1 range
            df['energy'] = MinMaxScaler().fit_transform(df[['energy']])

    # Make sure danceability is in 0-1 range
    if 'danceability' in df.columns:
        if df['danceability'].max() > 1.5:
            print("Normalizing danceability values...")
            df['danceability'] = df['danceability'] / df['danceability'].max()
    
    # Now create time_of_day based on audio features
    print("Creating time_of_day classification based on audio features...")
    
    # Method: Use K-means clustering to group songs into 4 clusters based on energy, danceability, BPM
    # Then map these clusters to time periods
    
    # Prepare features for clustering
    features_for_clustering = ['danceability', 'BPM']
    if 'energy' in df.columns:
        features_for_clustering.append('energy')
    
    # Create a copy of the dataframe with just the features we need
    clustering_df = df[features_for_clustering].copy()
    
    # Handle any NaN values
    clustering_df = clustering_df.fillna(clustering_df.mean())
    
    # Normalize the features for clustering
    normalized_features = MinMaxScaler().fit_transform(clustering_df)
    
    # Apply K-means clustering to identify 4 clusters (representing time periods)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(normalized_features)
    
    # Analyze the clusters to map them to time periods
    cluster_centers = pd.DataFrame(
        kmeans.cluster_centers_, 
        columns=features_for_clustering
    )
    
    print("Cluster centers:")
    print(cluster_centers)
    
    # Calculate "activity level" for each cluster (combination of energy and danceability)
    if 'energy' in cluster_centers.columns:
        cluster_centers['activity'] = (cluster_centers['energy'] + cluster_centers['danceability'] + 
                                      cluster_centers['BPM']/200)/3
    else:
        cluster_centers['activity'] = (cluster_centers['danceability'] + cluster_centers['BPM']/200)/2
    
    # Sort clusters by activity level
    cluster_activity = cluster_centers['activity'].sort_values()
    
    # Map clusters to time periods based on activity level
    time_mapping = {}
    for i, (cluster, _) in enumerate(cluster_activity.items()):
        if i == 0:
            time_mapping[cluster] = 'Morning (6-12)'    # Lowest activity -> Morning
        elif i == 1:
            time_mapping[cluster] = 'Night (0-6)'       # Second lowest -> Night
        elif i == 2:
            time_mapping[cluster] = 'Afternoon (12-18)' # Second highest -> Afternoon
        else:
            time_mapping[cluster] = 'Evening (18-24)'   # Highest activity -> Evening
    
    # Apply the mapping to create time_of_day column
    df['time_of_day'] = df['cluster'].map(time_mapping)
    
    print("Time of day distribution:")
    print(df['time_of_day'].value_counts())
    
    # Check if we have all necessary features now
    required_features = ['BPM', 'energy', 'danceability', 'time_of_day']
    missing_features = [feat for feat in required_features if feat not in df.columns]
    
    if missing_features:
        print(f"Warning: The following required features are still missing: {missing_features}")
        print("Available columns:", df.columns.tolist())
    else:
        # Set style for the plots
        plt.style.use('dark_background')
        
        # Create a scatterplot similar to Every Noise At Once

        # 1. Select features to use as x and y coordinates
        # We'll use energy (x-axis) and danceability (y-axis)
        x_feature = 'energy'
        y_feature = 'danceability'
        
        # 2. Process the time of day
        tod_order = ['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)']
        df['time_of_day'] = pd.Categorical(df['time_of_day'], categories=tod_order, ordered=True)
        
        # 3. Map time of day to color scheme
        time_colors = {
            'Night (0-6)': '#3b0066',      # Dark purple
            'Morning (6-12)': '#0099cc',   # Blue
            'Afternoon (12-18)': '#ffcc00', # Yellow
            'Evening (18-24)': '#ff4d00'    # Orange/Red
        }
        
        # 4. Map BPM to size
        # Normalize BPM to reasonable point sizes
        bpm_sizes = MinMaxScaler(feature_range=(20, 150)).fit_transform(df[['BPM']])
        
        # 5. Create the main scatter plot - Every Noise style
        plt.figure(figsize=(16, 12), facecolor='black')
        
        # First, add a light grid
        plt.grid(color='#333333', linestyle='-', linewidth=0.3, alpha=0.5)
        
        # Create an alpha channel
        alphas = [0.7] * len(df)
            
        # Plot each time period separately to maintain color control
        for time_period in df['time_of_day'].cat.categories:
            subset = df[df['time_of_day'] == time_period]
            
            # Skip if no songs in this time period
            if len(subset) == 0:
                continue
                
            # Get the corresponding sizes and alphas for this subset
            subset_sizes = bpm_sizes[df['time_of_day'] == time_period]
            subset_alphas = [alphas[i] for i, is_period in enumerate(df['time_of_day'] == time_period) if is_period]
            
            # Plot the points
            plt.scatter(
                subset[x_feature], 
                subset[y_feature],
                s=subset_sizes,
                c=[time_colors[time_period]], 
                alpha=subset_alphas,
                edgecolor='none',
                label=time_period
            )
            
            # Add song titles as text labels (just a sample)
            if len(subset) > 0:
                sample_size = min(5, len(subset))
                sample_indices = np.random.choice(subset.index, size=sample_size, replace=False)
                
                for idx in sample_indices:
                    x, y = subset.loc[idx, x_feature], subset.loc[idx, y_feature]
                    
                    # Get the song title
                    if 'title' in subset.columns:
                        label = subset.loc[idx, 'title']
                    else:
                        label = f"Song {idx}"
                    
                    # Add text with a subtle glow effect
                    txt = plt.text(
                        x, y + 0.02, 
                        label, 
                        color=time_colors[time_period],
                        fontsize=9, 
                        alpha=0.9,
                        ha='center'
                    )
                    # Add path effect for better visibility
                    txt.set_path_effects([
                        PathEffects.withStroke(linewidth=2, foreground='black')
                    ])
        
        # Add axis labels and title
        plt.xlabel('Energy (low → high)', fontsize=14)
        plt.ylabel('Danceability (low → high)', fontsize=14)
        plt.title('Every Song At Once: Music by Time of Day', fontsize=18)
        
        # Add a legend
        legend = plt.legend(title="Time of Day", fontsize=12, loc='upper right')
        legend.get_title().set_fontsize(14)
        
        # Customize the axes
        plt.xlim(-0.05, 1.05)
        plt.ylim(-0.05, 1.05)
        
        # Add descriptive text in corners
        plt.text(0.02, 0.02, 'LOW ENERGY', fontsize=10, color='#888888', alpha=0.7)
        plt.text(0.98, 0.02, 'HIGH ENERGY', fontsize=10, color='#888888', alpha=0.7, ha='right')
        plt.text(0.02, 0.98, 'LOW DANCEABILITY', fontsize=10, color='#888888', alpha=0.7, va='top')
        plt.text(0.98, 0.98, 'HIGH DANCEABILITY', fontsize=10, color='#888888', alpha=0.7, ha='right', va='top')
        
        plt.tight_layout()
        plt.show()
        
        # Create a second visualization to show time progression more explicitly
        plt.figure(figsize=(18, 10), facecolor='black')
        
        # Define our time zones clearly along the x-axis
        time_positions = {
            'Night (0-6)': 0, 
            'Morning (6-12)': 1, 
            'Afternoon (12-18)': 2, 
            'Evening (18-24)': 3
        }
        
        # Calculate a jitter to spread points horizontally within their time zone
        df['x_pos'] = df['time_of_day'].map(time_positions) + np.random.uniform(-0.3, 0.3, size=len(df))
        
        # Use danceability for y-axis and energy for color intensity
        scatter = plt.scatter(
            df['x_pos'],
            df['danceability'],
            s=bpm_sizes.flatten() * 0.8,  # Slightly smaller than the main plot
            c=df['energy'],
            cmap='plasma',
            alpha=0.7,
            edgecolor='none'
        )
        
        # Add a colorbar for energy
        cbar = plt.colorbar(scatter)
        cbar.set_label('Energy Level', fontsize=12)
        
        # Set the x-ticks to our time periods
        plt.xticks([0, 1, 2, 3], ['Night (0-6)', 'Morning (6-12)', 'Afternoon (12-18)', 'Evening (18-24)'])
        
        # Label the axes
        plt.xlabel('Time of Day', fontsize=14)
        plt.ylabel('Danceability', fontsize=14)
        plt.title('Song Characteristics Throughout the Day', fontsize=18)
        
        # Add a size legend for BPM
        handles, labels = [], []
        bpm_values = [80, 100, 120, 140]
        sizes = MinMaxScaler(feature_range=(20, 150)).fit_transform([[x] for x in bpm_values])
        
        for bpm, size in zip(bpm_values, sizes):
            handles.append(plt.scatter([], [], s=size, color='white', alpha=0.7))
            labels.append(f'{bpm} BPM')
        
        plt.legend(handles, labels, title="BPM (Tempo)", loc='upper right', title_fontsize=12)
        
        # Add grid for better readability
        plt.grid(color='#333333', linestyle='-', linewidth=0.3, alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        # Create another visualization that shows the "journey" of music throughout the day
        plt.figure(figsize=(14, 10), facecolor='black')
        
        # Calculate averages for each time period
        avg_by_time = df.groupby('time_of_day').agg({
            'energy': 'mean',
            'danceability': 'mean',
            'BPM': 'mean'
        }).reset_index()
        
        # Sort by the time order
        avg_by_time['time_of_day'] = pd.Categorical(
            avg_by_time['time_of_day'], 
            categories=tod_order,
            ordered=True
        )
        avg_by_time = avg_by_time.sort_values('time_of_day')
        
        # Plot the journey line
        plt.plot(
            avg_by_time['energy'], 
            avg_by_time['danceability'], 
            '-o', 
            color='white', 
            linewidth=2,
            markersize=10,
            alpha=0.8
        )
        
        # Add time labels to each point
        for i, row in avg_by_time.iterrows():
            txt = plt.text(
                row['energy'] + 0.02, 
                row['danceability'] + 0.02, 
                row['time_of_day'],
                color=time_colors[row['time_of_day']],
                fontsize=12,
                weight='bold'
            )
            txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='black')])
        
        # Add arrows to show direction
        for i in range(len(avg_by_time) - 1):
            plt.arrow(
                avg_by_time.iloc[i]['energy'],
                avg_by_time.iloc[i]['danceability'],
                (avg_by_time.iloc[i+1]['energy'] - avg_by_time.iloc[i]['energy']) * 0.8,
                (avg_by_time.iloc[i+1]['danceability'] - avg_by_time.iloc[i]['danceability']) * 0.8,
                head_width=0.02,
                head_length=0.02,
                fc='white',
                ec='white',
                alpha=0.6
            )
            
        # Add contours or density estimation for each time period
        for time_period in avg_by_time['time_of_day']:
            subset = df[df['time_of_day'] == time_period]
            
            if len(subset) >= 10:  # Need enough points for kernel density
                try:
                    # Create kernel density estimate
                    sns.kdeplot(
                        x=subset['energy'],
                        y=subset['danceability'],
                        levels=3,
                        color=time_colors[time_period],
                        alpha=0.4,
                        linewidths=1
                    )
                except Exception as e:
                    print(f"Could not create density plot for {time_period}: {e}")
            
        plt.xlabel('Energy', fontsize=14)
        plt.ylabel('Danceability', fontsize=14)
        plt.title('Music Journey Throughout the Day', fontsize=18)
        
        # Set axis limits
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        
        # Add grid
        plt.grid(color='#333333', linestyle='-', linewidth=0.3, alpha=0.5)
        
        plt.tight_layout()
        plt.show()
        
        print("\nVisualizations complete!")