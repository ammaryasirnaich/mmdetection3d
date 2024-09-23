import plotly.graph_objects as go
import pandas as pd

# Data for the plot
data = {
    "Model": ["TED-S (TeSpConv)", "M3DETR", "CasA-V", "CT3D", "ours"],
    "Params (M)": [9.30, 19.65, 11.30, 7.85, 11.7],
    "FPS": [5.6, 4, 6.4, 7.5, 15],
    "mAP (%)": [85.81, 76.96, 80.08, 83.46, 70.24]
}

# Create a dataframe
df = pd.DataFrame(data)

# Create the figure
fig = go.Figure()

# Bar plot for Params (M) - left y-axis
fig.add_trace(go.Bar(
    x=df["Model"],
    y=df["Params (M)"],
    name="Params (M)",
    marker_color="skyblue",
    yaxis="y1"  # Left y-axis
))

# Line plot for FPS - right y-axis
fig.add_trace(go.Scatter(
    x=df["Model"],
    y=df["FPS"],
    mode="lines+markers",
    name="FPS",
    marker=dict(color="green", size=10),
    line=dict(color="green"),
    yaxis="y2"  # Right y-axis
))

# Line plot for mAP (%) - right y-axis
fig.add_trace(go.Scatter(
    x=df["Model"],
    y=df["mAP (%)"],
    mode="lines+markers",
    name="mAP (%)",
    marker=dict(color="orange", size=10),
    line=dict(color="orange"),
    yaxis="y2"  # Right y-axis
))

# Update the layout with dual y-axes and adjust the legend
fig.update_layout(
    title="Comparison of Model Parameters, FPS, and mAP",
    xaxis=dict(title="Model"),
    yaxis=dict(
        title="Params (M)",
        titlefont=dict(color="skyblue"),
        tickfont=dict(color="skyblue")
    ),
    yaxis2=dict(
        title="FPS / mAP (%)",
        titlefont=dict(color="orange"),
        tickfont=dict(color="orange"),
        overlaying="y",  # Overlay on the same graph
        side="right"     # Show on the right side
    ),
    legend=dict(
        x=0.80,  # Adjust x position (move inside)
        y=1,     # Adjust y position (move inside)
        bgcolor="rgba(255,255,255,0)",
        bordercolor="Black",
        borderwidth=1
    ),
    width=800,  # Adjust width
    height=600  # Adjust height
)

# Show the plot
fig.show()
