"""
Wall-Following Robot Navigation using MDP
Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import json
import os

st.set_page_config(
    page_title="MDP Robot Navigation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    data = {}
    data_paths = ['data/', '']
    
    for path in data_paths:
        try:
            data['values'] = pd.read_csv(f'{path}optimal_value_function.csv')
            data['q_values'] = pd.read_csv(f'{path}q_values.csv')
            data['transitions'] = pd.read_csv(f'{path}transitions.csv')
            data['rewards'] = pd.read_csv(f'{path}rewards.csv')
            data['convergence'] = pd.read_csv(f'{path}convergence.csv')
            data['trajectory'] = pd.read_csv(f'{path}trajectory.csv')
            
            with open(f'{path}mdp_info.json', 'r') as f:
                data['info'] = json.load(f)
            
            data['path'] = path
            return data
        except FileNotFoundError:
            continue
    
    return None

def get_image_path(filename):
    paths = [f'images/{filename}', filename]
    for p in paths:
        if os.path.exists(p):
            return p
    return None

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
        st.metric("Total States", data['info']['num_states'])
        st.metric("Total Actions", data['info']['num_actions'])
        st.metric("Discount Factor (γ)", data['info']['gamma'])
        st.metric("Iterations", data['info']['convergence_iterations'])
        st.metric("Mean V(s)", f"{data['info']['mean_value']:.2f}")
        st.metric("Max V(s)", f"{data['info']['max_value']:.2f}")
    
    st.markdown("---")
    st.subheader("📊 Dataset Exploration")
    img_path = get_image_path('01_data_exploration.png')
    if img_path:
        st.image(img_path, use_column_width=True)

def show_mdp_components(data):
    st.header("📊 MDP Components")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔵 States", "🟢 Actions", "🔄 Transitions", "🎁 Rewards"])
    
    with tab1:
        st.subheader("State Space")
        st.markdown("""
        States are discretized sensor readings: **Front_Left_Right_Back**
        
        Each sensor value is categorized as:
        - **VC** = Very Close
        - **C** = Close
        - **M** = Medium
        - **F** = Far
        """)
        
        states = data['info']['states']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total States", len(states))
        with col2:
            st.metric("State Format", "F_L_R_B")
        with col3:
            st.metric("Categories", "VC, C, M, F")
        
        st.dataframe(pd.DataFrame({'State': states[:20]}), height=300)
    
    with tab2:
        st.subheader("Action Space")
        actions = data['info']['actions']
        
        action_desc = {
            'Move-Forward': '⬆️ Move one step in current direction',
            'Slight-Right-Turn': '↪️ Turn slightly right',
            'Sharp-Right-Turn': '⤵️ Turn right and move forward',
            'Slight-Left-Turn': '↩️ Turn slightly left'
        }
        
        for action in actions:
            st.markdown(f"**{action}**: {action_desc.get(action, 'Movement action')}")
        
        policy_dist = data['values']['Optimal_Policy'].value_counts()
        fig = px.pie(values=policy_dist.values, names=policy_dist.index, 
                     title="Actions in Optimal Policy")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Transition Probabilities P(s'|s,a)")
        st.metric("Total Transitions", len(data['transitions']))
        st.dataframe(data['transitions'].head(20), height=400)
        
        fig = px.histogram(data['transitions'], x='Probability', nbins=30,
                          title="Distribution of Transition Probabilities")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Reward Function R(s,a)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Reward", f"{data['rewards']['Reward'].mean():.2f}")
        with col2:
            st.metric("Max Reward", f"{data['rewards']['Reward'].max():.2f}")
        with col3:
            st.metric("Min Reward", f"{data['rewards']['Reward'].min():.2f}")
        
        reward_by_action = data['rewards'].groupby('Action')['Reward'].mean().sort_values(ascending=False)
        fig = px.bar(x=reward_by_action.index, y=reward_by_action.values,
                    title="Average Reward per Action",
                    labels={'x': 'Action', 'y': 'Average Reward'})
        st.plotly_chart(fig, use_container_width=True)

def show_value_iteration(data):
    st.header("🔄 Value Iteration Results")
    
    with st.expander("📖 Value Iteration Algorithm"):
        st.markdown("""
        ### Bellman Optimality Equation
        
        Value Iteration computes the optimal value function by iteratively applying the Bellman equation.
        
        **Parameters:**
        - γ (gamma) = 0.9 (discount factor)
        - θ = 1e-6 (convergence threshold)
        - Max iterations = 1000
        """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📉 Convergence History")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['convergence']['Iteration'],
            y=data['convergence']['Delta'],
            mode='lines',
            name='Max Delta',
            line=dict(color='blue', width=2)
        ))
        fig.add_hline(y=1e-6, line_dash="dash", line_color="red")
        fig.update_layout(
            title="Value Iteration Convergence",
            xaxis_title="Iteration",
            yaxis_title="Max Value Change",
            yaxis_type="log"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Value Function Distribution")
        fig = px.histogram(data['values'], x='Optimal_Value', nbins=30,
                          title="Distribution of State Values V(s)")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("📈 Complete MDP Results")
    img_path = get_image_path('02_mdp_results.png')
    if img_path:
        st.image(img_path, use_column_width=True)
    
    st.subheader("📋 Optimal Value Function & Policy")
    sort_by = st.selectbox("Sort by:", ["Optimal_Value (High to Low)", "Optimal_Value (Low to High)", "State"])
    
    df_display = data['values'].copy()
    if sort_by == "Optimal_Value (High to Low)":
        df_display = df_display.sort_values('Optimal_Value', ascending=False)
    elif sort_by == "Optimal_Value (Low to High)":
        df_display = df_display.sort_values('Optimal_Value', ascending=True)
    else:
        df_display = df_display.sort_values('State')
    
    st.dataframe(df_display, height=400, use_container_width=True)

def show_simulation(data):
    st.header("🎮 Robot Simulation")
    
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
    
    tab1, tab2, tab3, tab4 = st.tabs(["🗺️ Movement Grid", "🖼️ Step Frames", "📊 Trajectory", "🔥 Heatmap"])
    
    with tab1:
        st.subheader("Agent's Movement in Grid")
        img_path = get_image_path('03_agent_movement_grid.png')
        if img_path:
            st.image(img_path, use_column_width=True)
    
    with tab2:
        st.subheader("Step-by-Step Movement Frames")
        img_path = get_image_path('05_movement_frames.png')
        if img_path:
            st.image(img_path, use_column_width=True)
    
    with tab3:
        st.subheader("Trajectory Analysis")
        img_path = get_image_path('06_trajectory_analysis.png')
        if img_path:
            st.image(img_path, use_column_width=True)
    
    with tab4:
        st.subheader("Visit Frequency Heatmap")
        img_path = get_image_path('04_room_navigation.png')
        if img_path:
            st.image(img_path, use_column_width=True)
    
    st.subheader("📋 Trajectory Data")
    st.dataframe(data['trajectory'].head(50), height=300, use_container_width=True)

def show_analysis(data):
    st.header("📈 Detailed Analysis")
    
    tab1, tab2, tab3 = st.tabs(["🎯 Policy Analysis", "📊 State Analysis", "🔄 Action Analysis"])
    
    with tab1:
        st.subheader("Optimal Policy Analysis")
        policy_counts = data['values']['Optimal_Policy'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(values=policy_counts.values, names=policy_counts.index,
                        title="Optimal Policy Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(x=policy_counts.index, y=policy_counts.values,
                        title="Actions in Optimal Policy",
                        labels={'x': 'Action', 'y': 'Number of States'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("State Value Analysis")
        
        df = data['values'].copy()
        df['Front'] = df['State'].apply(lambda x: x.split('_')[0])
        
        fig = px.box(df, x='Front', y='Optimal_Value', 
                    title="State Value by Front Sensor")
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 🏆 Top 10 Highest Value States")
            st.dataframe(data['values'].nlargest(10, 'Optimal_Value'), use_container_width=True)
        with col2:
            st.markdown("#### ⚠️ Top 10 Lowest Value States")
            st.dataframe(data['values'].nsmallest(10, 'Optimal_Value'), use_container_width=True)
    
    with tab3:
        st.subheader("Action Analysis in Simulation")
        action_counts = data['trajectory']['action'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.pie(values=action_counts.values, names=action_counts.index,
                        title="Actions Taken During Simulation")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.box(data['trajectory'], x='action', y='value',
                        title="State Value by Action Taken")
            st.plotly_chart(fig, use_container_width=True)

def show_downloads(data):
    st.header("📁 Download Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 CSV Files")
        
        st.download_button(
            label="📥 Optimal Value Function",
            data=data['values'].to_csv(index=False),
            file_name="optimal_value_function.csv",
            mime="text/csv"
        )
        
        st.download_button(
            label="📥 Q-Values",
            data=data['q_values'].to_csv(index=False),
            file_name="q_values.csv",
            mime="text/csv"
        )
        
        st.download_button(
            label="📥 Transitions",
            data=data['transitions'].to_csv(index=False),
            file_name="transitions.csv",
            mime="text/csv"
        )
        
        st.download_button(
            label="📥 Rewards",
            data=data['rewards'].to_csv(index=False),
            file_name="rewards.csv",
            mime="text/csv"
        )
        
        st.download_button(
            label="📥 Trajectory",
            data=data['trajectory'].to_csv(index=False),
            file_name="trajectory.csv",
            mime="text/csv"
        )
    
    with col2:
        st.subheader("📋 JSON Files")
        
        st.download_button(
            label="📥 MDP Info",
            data=json.dumps(data['info'], indent=2),
            file_name="mdp_info.json",
            mime="application/json"
        )
        
        st.subheader("📊 Data Preview")
        file_choice = st.selectbox("Preview:", ["Values", "Q-Values", "Transitions", "Rewards", "Trajectory"])
        
        file_map = {
            "Values": data['values'],
            "Q-Values": data['q_values'],
            "Transitions": data['transitions'],
            "Rewards": data['rewards'],
            "Trajectory": data['trajectory']
        }
        st.dataframe(file_map[file_choice].head(20), use_container_width=True)

def main():
    st.markdown('<p class="main-header">🤖 Wall-Following Robot Navigation</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Markov Decision Process with Value Iteration</p>', unsafe_allow_html=True)
    
    data = load_data()
    
    if data is None:
        st.error("⚠️ Data files not found! Please ensure CSV files are in the 'data/' folder.")
        return
    
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
        st.metric("Iterations", data['info']['convergence_iterations'])
    
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

if __name__ == "__main__":
    main()
