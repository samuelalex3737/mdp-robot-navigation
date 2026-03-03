---

## File 3: `app.py` (Complete Streamlit App)

```python
"""
Wall-Following Robot Navigation using MDP
Streamlit Application

Author: [Your Name]
Course: [Your Course Name]
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MDP Robot Navigation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# LOAD DATA FUNCTIONS
# ============================================================

@st.cache_data
def load_data():
    """Load all data files"""
    data = {}
    
    # Try to load from data folder first, then current directory
    data_paths = ['data/', '']
    
    for path in data_paths:
        try:
            data['values'] = pd.read_csv(f'{path}optimal_value_function.csv')
            data['q_values'] = pd.read_csv(f'{path}q_values.csv')
            data['transitions'] = pd.read_csv(f'{path}transitions.csv')
            data['rewards'] = pd.read_csv(f'{path}rewards.csv')
            data['convergence'] = pd.read_csv(f'{path}convergence.csv')
            data['trajectory'] = pd.read_csv(f'{path}trajectory.csv')
            
            # Load JSON
            with open(f'{path}mdp_info.json', 'r') as f:
                data['info'] = json.load(f)
            
            data['path'] = path
            return data
        except FileNotFoundError:
            continue
    
    return None

@st.cache_data
def get_image_path(filename):
    """Get correct path for images"""
    paths = [f'images/{filename}', filename]
    for p in paths:
        if os.path.exists(p):
            return p
    return None

# ============================================================
# MAIN APP
# ============================================================

def main():
    # Header
    st.markdown('<p class="main-header">🤖 Wall-Following Robot Navigation</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Markov Decision Process with Value Iteration</p>', unsafe_allow_html=True)
    
    # Load data
    data = load_data()
    
    if data is None:
        st.error("⚠️ Data files not found! Please ensure CSV files are in the 'data/' folder.")
        st.info("Required files: optimal_value_function.csv, q_values.csv, transitions.csv, rewards.csv, convergence.csv, trajectory.csv, mdp_info.json")
        return
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/robot-2.png", width=80)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page:",
            ["🏠 Overview", "📊 MDP Components", "🔄 Value Iteration", 
             "🎮 Robot Simulation", "📈 Analysis", "📁 Download Data"]
        )
        
        st.markdown("---")
        st.markdown("### 📌 Quick Stats")
        st.metric("States", data['info']['num_states'])
        st.metric("Actions", data['info']['num_actions'])
        st.metric("Convergence", f"{data['info']['convergence_iterations']} iter")
    
    # Page routing
    if page == "🏠 Overview":
        show_overview(data)
    elif page == "📊 MDP Components":
        show_mdp_components(data)
    elif page == "🔄 Value Iteration":
        show_value_iteration(data)
    elif page == "🎮 Robot Simulation":
        show_simulation(data)
    elif page == "📈 Analysis":
        show_analysis(data)
    elif page == "📁 Download Data":
        show_downloads(data)

# ============================================================
# PAGE: OVERVIEW
# ============================================================

def show_overview(data):
    st.header("📋 Project Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Problem Statement
        
        A mobile robot equipped with **ultrasonic sensors** navigates a room while following walls. 
        The robot must learn an **optimal policy** to:
        
        - ✅ Follow walls safely
        - ✅ Avoid collisions
        - ✅ Navigate efficiently
        
        ### Approach
        
        We model this as a **Markov Decision Process (MDP)** and solve it using **Value Iteration**.
        
        ### Dataset
        
        - **Source:** Kaggle Wall-Following Robot Dataset
        - **Sensors:** 4 ultrasonic sensors (Front, Left, Right, Back)
        - **Actions:** Move-Forward, Slight-Right-Turn, Sharp-Right-Turn, Slight-Left-Turn
        """)
    
    with col2:
        st.markdown("### Key Metrics")
        
        metrics = [
            ("Total States", data['info']['num_states']),
            ("Total Actions", data['info']['num_actions']),
            ("Discount Factor (γ)", data['info']['gamma']),
            ("Iterations", data['info']['convergence_iterations']),
            ("Mean V(s)", f"{data['info']['mean_value']:.2f}"),
            ("Max V(s)", f"{data['info']['max_value']:.2f}"),
        ]
        
        for name, value in metrics:
            st.metric(name, value)
    
    st.markdown("---")
    
    # Show data exploration image
    st.subheader("📊 Dataset Exploration")
    img_path = get_image_path('01_data_exploration.png')
    if img_path:
        st.image(img_path, use_column_width=True)
    else:
        st.info("Image not found: 01_data_exploration.png")

# ============================================================
# PAGE: MDP COMPONENTS
# ============================================================

def show_mdp_components(data):
    st.header("📊 MDP Components")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔵 States", "🟢 Actions", "🔄 Transitions", "🎁 Rewards"])
    
    with tab1:
        st.subheader("State Space")
        st.markdown("""
        States are discretized sensor readings: **Front_Left_Right_Back**
        
        Each sensor value is categorized as:
        - **VC** = Very Close (≤1 unit)
        - **C** = Close (≤2 units)
        - **M** = Medium (≤4 units)
        - **F** = Far (>4 units)
        """)
        
        states = data['info']['states']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total States", len(states))
        with col2:
            st.metric("State Format", "F_L_R_B")
        with col3:
            st.metric("Categories", "VC, C, M, F")
        
        st.markdown("#### Sample States")
        st.dataframe(pd.DataFrame({'State': states[:20]}), height=300)
    
    with tab2:
        st.subheader("Action Space")
        
        actions = data['info']['actions']
        
        action_desc = {
            'Move-Forward': '⬆️ Move one step in current direction',
            'Slight-Right-Turn': '↪️ Turn slightly right (45-90°)',
            'Sharp-Right-Turn': '⤵️ Turn right and move forward',
            'Slight-Left-Turn': '↩️ Turn slightly left (45-90°)'
        }
        
        for action in actions:
            st.markdown(f"**{action}**: {action_desc.get(action, 'Movement action')}")
        
        # Action distribution in policy
        st.markdown("#### Optimal Policy Action Distribution")
        policy_dist = data['values']['Optimal_Policy'].value_counts()
        
        fig = px.pie(values=policy_dist.values, names=policy_dist.index, 
                     title="Actions in Optimal Policy",
                     color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Transition Probabilities P(s'|s,a)")
        
        st.markdown("""
        Transition probabilities are learned from the dataset by counting 
        how often each state-action pair leads to each next state.
        """)
        
        st.metric("Total Transitions", len(data['transitions']))
        
        # Show sample transitions
        st.markdown("#### Sample Transitions")
        st.dataframe(data['transitions'].head(20), height=400)
        
        # Transition visualization
        st.markdown("#### Transition Probability Distribution")
        fig = px.histogram(data['transitions'], x='Probability', nbins=30,
                          title="Distribution of Transition Probabilities")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Reward Function R(s,a)")
        
        st.markdown("""
        Rewards are designed to encourage wall-following behavior:
        
        - ✅ **Positive reward** for moving forward when safe
        - ✅ **Positive reward** for turning when wall is close
        - ❌ **Negative reward** for moving toward walls
        - ❌ **Small step cost** for efficiency
        """)
        
        # Reward statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Reward", f"{data['rewards']['Reward'].mean():.2f}")
        with col2:
            st.metric("Max Reward", f"{data['rewards']['Reward'].max():.2f}")
        with col3:
            st.metric("Min Reward", f"{data['rewards']['Reward'].min():.2f}")
        
        # Reward heatmap by action
        st.markdown("#### Average Reward by Action")
        reward_by_action = data['rewards'].groupby('Action')['Reward'].mean().sort_values(ascending=False)
        
        fig = px.bar(x=reward_by_action.index, y=reward_by_action.values,
                    title="Average Reward per Action",
                    labels={'x': 'Action', 'y': 'Average Reward'},
                    color=reward_by_action.values,
                    color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE: VALUE ITERATION
# ============================================================

def show_value_iteration(data):
    st.header("🔄 Value Iteration Results")
    
    # Algorithm explanation
    with st.expander("📖 Value Iteration Algorithm", expanded=False):
        st.markdown("""
        ### Bellman Optimality Equation
        
        Value Iteration computes the optimal value function by iteratively applying:
        
        $$V_{k+1}(s) = \max_a \left[ R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s') \\right]$$
        
        **Parameters:**
        - γ (gamma) = 0.9 (discount factor)
        - θ = 1e-6 (convergence threshold)
        - Max iterations = 1000
        
        **Convergence:** Algorithm stops when max|V_{k+1}(s) - V_k(s)| < θ
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Convergence plot
        st.subheader("📉 Convergence History")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['convergence']['Iteration'],
            y=data['convergence']['Delta'],
            mode='lines',
            name='Max Delta',
            line=dict(color='blue', width=2)
        ))
        fig.add_hline(y=1e-6, line_dash="dash", line_color="red", 
                      annotation_text="Convergence Threshold")
        fig.update_layout(
            title="Value Iteration Convergence",
            xaxis_title="Iteration",
            yaxis_title="Max Value Change (Δ)",
            yaxis_type="log"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Value distribution
        st.subheader("📊 Value Function Distribution")
        
        fig = px.histogram(data['values'], x='Optimal_Value', nbins=30,
                          title="Distribution of State Values V(s)",
                          color_discrete_sequence=['green'])
        fig.add_vline(x=data['info']['mean_value'], line_dash="dash", 
                      annotation_text=f"Mean: {data['info']['mean_value']:.2f}")
        st.plotly_chart(fig, use_container_width=True)
    
    # Results image
    st.subheader("📈 Complete MDP Results")
    img_path = get_image_path('02_mdp_results.png')
    if img_path:
        st.image(img_path, use_column_width=True)
    
    # Value function table
    st.subheader("📋 Optimal Value Function & Policy")
    
    # Sort options
    sort_by = st.selectbox("Sort by:", ["Optimal_Value (High to Low)", "Optimal_Value (Low to High)", "State"])
    
    df_display = data['values'].copy()
    if sort_by == "Optimal_Value (High to Low)":
        df_display = df_display.sort_values('Optimal_Value', ascending=False)
    elif sort_by == "Optimal_Value (Low to High)":
        df_display = df_display.sort_values('Optimal_Value', ascending=True)
    else:
        df_display = df_display.sort_values('State')
    
    st.dataframe(df_display, height=400, use_container_width=True)

# ============================================================
# PAGE: ROBOT SIMULATION
# ============================================================

def show_simulation(data):
    st.header("🎮 Robot Simulation")
    
    st.markdown("""
    The robot follows the **right-hand wall following rule** while using the learned MDP policy 
    for state value estimation.
    """)
    
    # Simulation stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Steps", len(data['trajectory']))
    with col2:
        unique_pos = data['trajectory']['position'].nunique()
        st.metric("Unique Positions", unique_pos)
    with col3:
        coverage = data['info'].get('room_coverage_percent', 0)
        st.metric("Room Coverage", f"{coverage:.1f}%")
    with col4:
        mean_v = data['trajectory']['value'].mean()
        st.metric("Mean V(s)", f"{mean_v:.2f}")
    
    # Images
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Movement Grid", "🖼️ Step Frames", "📊 Trajectory", "🔥 Heatmap"])
    
    with tab1:
        st.subheader("Agent's Movement in Grid")
        img_path = get_image_path('03_agent_movement_grid.png')
        if img_path:
            st.image(img_path, use_column_width=True)
        else:
            st.info("Image not found")
    
    with tab2:
        st.subheader("Step-by-Step Movement Frames")
        img_path = get_image_path('05_movement_frames.png')
        if img_path:
            st.image(img_path, use_column_width=True)
        else:
            st.info("Image not found")
    
    with tab3:
        st.subheader("Trajectory Analysis")
        img_path = get_image_path('06_trajectory_analysis.png')
        if img_path:
            st.image(img_path, use_column_width=True)
        else:
            st.info("Image not found")
    
    with tab4:
        st.subheader("Visit Frequency Heatmap")
        img_path = get_image_path('04_room_navigation.png')
        if img_path:
            st.image(img_path, use_column_width=True)
        else:
            st.info("Image not found")
    
    # Interactive trajectory table
    st.subheader("📋 Trajectory Data")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        step_range = st.slider("Step Range:", 0, len(data['trajectory'])-1, (0, min(50, len(data['trajectory'])-1)))
    with col2:
        action_filter = st.multiselect("Filter by Action:", data['trajectory']['action'].unique(), 
                                       default=data['trajectory']['action'].unique())
    
    filtered_traj = data['trajectory'][
        (data['trajectory']['step'] >= step_range[0]) & 
        (data['trajectory']['step'] <= step_range[1]) &
        (data['trajectory']['action'].isin(action_filter))
    ]
    
    st.dataframe(filtered_traj, height=300, use_container_width=True)

# ============================================================
# PAGE: ANALYSIS
# ============================================================

def show_analysis(data):
    st.header("📈 Detailed Analysis")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Policy Analysis", "📊 State Analysis", "🔄 Action Analysis"])
    
    with tab1:
        st.subheader("Optimal Policy Analysis")
        
        # Policy distribution
        policy_counts = data['values']['Optimal_Policy'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=policy_counts.values, names=policy_counts.index,
                        title="Optimal Policy Distribution",
                        color_discrete_sequence=px.colors.qualitative.Bold)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(x=policy_counts.index, y=policy_counts.values,
                        title="Actions in Optimal Policy",
                        labels={'x': 'Action', 'y': 'Number of States'},
                        color=policy_counts.values,
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        # Policy by state characteristics
        st.markdown("#### Policy by Front Sensor Reading")
        
        df = data['values'].copy()
        df['Front'] = df['State'].apply(lambda x: x.split('_')[0])
        
        policy_by_front = df.groupby(['Front', 'Optimal_Policy']).size().unstack(fill_value=0)
        
        fig = px.bar(policy_by_front, barmode='group',
                    title="Optimal Action by Front Sensor Reading",
                    labels={'value': 'Count', 'Front': 'Front Sensor'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("State Value Analysis")
        
        df = data['values'].copy()
        df['Front'] = df['State'].apply(lambda x: x.split('_')[0])
        df['Left'] = df['State'].apply(lambda x: x.split('_')[1])
        df['Right'] = df['State'].apply(lambda x: x.split('_')[2])
        df['Back'] = df['State'].apply(lambda x: x.split('_')[3])
        
        # Value by sensor
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(df, x='Front', y='Optimal_Value', 
                        title="State Value by Front Sensor",
                        category_orders={'Front': ['VC', 'C', 'M', 'F']})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, x='Right', y='Optimal_Value',
                        title="State Value by Right Sensor",
                        category_orders={'Right': ['VC', 'C', 'M', 'F']})
            st.plotly_chart(fig, use_container_width=True)
        
        # Top and bottom states
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🏆 Top 10 Highest Value States")
            st.dataframe(data['values'].nlargest(10, 'Optimal_Value'), use_container_width=True)
        
        with col2:
            st.markdown("#### ⚠️ Top 10 Lowest Value States")
            st.dataframe(data['values'].nsmallest(10, 'Optimal_Value'), use_container_width=True)
    
    with tab3:
        st.subheader("Action Analysis in Simulation")
        
        # Action distribution in trajectory
        action_counts = data['trajectory']['action'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=action_counts.values, names=action_counts.index,
                        title="Actions Taken During Simulation",
                        color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Value by action
            fig = px.box(data['trajectory'], x='action', y='value',
                        title="State Value by Action Taken",
                        color='action')
            st.plotly_chart(fig, use_container_width=True)
        
        # Action over time
        st.markdown("#### Actions Over Time")
        action_time = data['trajectory'].groupby(['step', 'action']).size().reset_index(name='count')
        
        fig = px.scatter(data['trajectory'], x='step', y='value', color='action',
                        title="State Values During Navigation (Colored by Action)",
                        labels={'step': 'Step', 'value': 'State Value V(s)'})
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE: DOWNLOADS
# ============================================================

def show_downloads(data):
    st.header("📁 Download Data")
    
    st.markdown("""
    Download all data files and visualizations from this project.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 CSV Files")
        
        # Optimal Value Function
        st.download_button(
            label="📥 Download Optimal Value Function",
            data=data['values'].to_csv(index=False),
            file_name="optimal_value_function.csv",
            mime="text/csv"
        )
        
        # Q-Values
        st.download_button(
            label="📥 Download Q-Values",
            data=data['q_values'].to_csv(index=False),
            file_name="q_values.csv",
            mime="text/csv"
        )
        
        # Transitions
        st.download_button(
            label="📥 Download Transitions",
            data=data['transitions'].to_csv(index=False),
            file_name="transitions.csv",
            mime="text/csv"
        )
        
        # Rewards
        st.download_button(
            label="📥 Download Rewards",
            data=data['rewards'].to_csv(index=False),
            file_name="rewards.csv",
            mime="text/csv"
        )
        
        # Convergence
        st.download_button(
            label="📥 Download Convergence History",
            data=data['convergence'].to_csv(index=False),
            file_name="convergence.csv",
            mime="text/csv"
        )
        
        # Trajectory
        st.download_button(
            label="📥 Download Trajectory",
            data=data['trajectory'].to_csv(index=False),
            file_name="trajectory.csv",
            mime="text/csv"
        )
    
    with col2:
        st.subheader("📋 JSON Files")
        
        st.download_button(
            label="📥 Download MDP Info",
            data=json.dumps(data['info'], indent=2),
            file_name="mdp_info.json",
            mime="application/json"
        )
        
        st.subheader("📊 Data Preview")
        
        file_choice = st.selectbox("Preview File:", 
                                   ["Optimal Value Function", "Q-Values", "Transitions", "Rewards", "Trajectory"])
        
        file_map = {
            "Optimal Value Function": data['values'],
            "Q-Values": data['q_values'],
            "Transitions": data['transitions'],
            "Rewards": data['rewards'],
            "Trajectory": data['trajectory']
        }
        
        st.dataframe(file_map[file_choice].head(20), use_container_width=True)

# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    main()