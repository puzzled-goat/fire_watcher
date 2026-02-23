import base64
import sys
from io import BytesIO

sys.path.append("../..")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html, no_update
from PIL import Image
from sklearn.cluster import KMeans

# based on: https://dash.plotly.com/dash-core-components/tooltip?_gl=1*c3xxsc*_gcl_au*NjY0MDA2NDUzLjE3Njk0MTY0MDY.*_ga*MTQ0NzY2MzYyOS4xNzY5NDE2NDA3*_ga_6G7EE0JNSC*czE3Njk0MTY0MDckbzEkZzEkdDE3Njk0MTc1NTMkajI1JGwwJGgw

ASSET_DIR = "latent_assets"
LATENTS = np.load(f"{ASSET_DIR}/autoencoder_vae_v2_latents.npy")
TSNE_2D = np.load(f"{ASSET_DIR}/autoencoder_vae_v2_tsne_2d.npy")
IMAGES = np.load(
    f"{ASSET_DIR}/autoencoder_vae_v2_images.npy", allow_pickle=True
)  # images as arrays


def downscale(img_array, size=(32, 32)):
    img = Image.fromarray((img_array * 255).astype("uint8").transpose(1, 2, 0))
    img = img.resize(size)
    return np.array(img).transpose(2, 0, 1) / 255.0


# Downscale images
IMAGES = np.array([downscale(img) for img in IMAGES])

# Limit the dataset size
max_points = 2000
TSNE_2D = TSNE_2D[:max_points]
IMAGES = IMAGES[:max_points]
LATENTS = LATENTS[:max_points]

# Apply KMeans clustering
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(LATENTS)
cluster_labels = kmeans.labels_


# Convert images to base64 for hover
def array_to_base64(img_array):
    img = Image.fromarray((img_array * 255).astype(np.uint8).transpose(1, 2, 0))
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


img_urls = [array_to_base64(IMAGES[i]) for i in range(len(IMAGES))]

# Prepare data for Plotly scatter plot
df = pd.DataFrame(
    {
        "x": TSNE_2D[:, 0],
        "y": TSNE_2D[:, 1],
        "cluster": cluster_labels.astype(str),
        "index": np.arange(len(TSNE_2D)),
        "image": img_urls,
    }
)

# Create color scale for clusters
color_map = {
    "0": "#E52B50",  # Red
    "1": "#9F2B68",  # Purple
    "2": "#3B7A57",  # Green
    "3": "#3DDC84",  # Light Green
    "4": "#FFBF00",  # Yellow
    # Add more colors if you have more clusters
}

# Assign colors based on clusters
colors = [color_map[str(label)] for label in cluster_labels]

# Create the Plotly scatter plot
fig = go.Figure(
    data=[
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers",
            marker=dict(size=6, color=colors, showscale=False),
        )
    ]
)

fig.update_layout(
    width=1200,
    height=1200,
    xaxis=dict(scaleanchor="y", scaleratio=1),
    yaxis=dict(constrain="domain"),
)


app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Variational Autoencoder Latent Space Explorer"),
        dcc.Graph(id="scatter-plot", figure=fig),
        # Hidden Div for Hover Data
        dcc.Tooltip(
            id="graph-tooltip",
            direction="bottom",
        ),
    ]
)


# Callback for hover behavior
@app.callback(
    Output("graph-tooltip", "show"),
    Output("graph-tooltip", "bbox"),
    Output("graph-tooltip", "children"),
    Input("scatter-plot", "hoverData"),
)
def display_hover(hoverData):
    if hoverData is None:
        return False, no_update, no_update

    hover_data = hoverData["points"][0]
    num = hover_data["pointNumber"]

    im_url = img_urls[num]
    children = [
        html.Div(
            [
                html.Img(
                    src=im_url,
                    style={"width": "50px", "display": "block", "margin": "0 auto"},
                ),
                html.P(f"Cluster {df['cluster'][num]}", style={"font-weight": "bold"}),
            ]
        )
    ]

    bbox = hover_data["bbox"]
    return True, bbox, children


if __name__ == "__main__":
    app.run(debug=True)
