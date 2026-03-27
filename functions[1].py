# Load json to parse JSON text for functions module
import json

# Load sys so Python can locate files in list of directories
import sys

# Load os to get the current directory path
import os

# load inspect to examine Python objects
import inspect

# Load geopandas library to work with geoJSON files.
import geopandas as gpd

# Load pandas library for data manipulation and analysis.
import pandas as pd

# Load matplotlib library for data visualization.
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Load folium to visualize geospatial data.
import folium

# Load cmocean for colormaps.
import cmocean.cm as cmo

# Load datetime to manipulate dates and times.
from datetime import datetime

# Load requests to conduct HTTP requests to an API.
import requests

# Load time so we can time and optimize queries.
import time

# Load Point to use in coordinate mapping.
from shapely.geometry import Point


# for cloropleth map function
from branca.element import MacroElement

# for cloropleth map function
from jinja2 import Template

#______________________________________________________________________________

def summarize_dataframe(df):
    """
    Summarizes a DataFrame/GeoDataFrame with column-level statistics.
    Args:
        df: a pandas DataFrame or geopandas GeoDataFrame to summarize
    Returns:
        DataFrame with summary statistics for each column:
            Type: column data type
            Missing: count of missing values
            Missing %: percent of missing values
            Unique: count of unique values
            min, max, mean: minimum, maximum, and mean (numeric columns only)
    Raises:
        AttributeError: If df input is not a DataFrame-like object
        ZeroDivisionError: If DataFrame is empty
    """
    summary = {
        col: {
            'Type': df[col].dtype,
            'Missing': df[col].isnull().sum(),
            'Missing %': round(df[col].isnull().sum() / len(df) * 100, 2),
            'Unique': df[col].nunique()
        }
        for col in df.columns
    }
    
    # Transpose summary so each row represents one column from df.
    summary_df = pd.DataFrame(summary).T

    # Grab numeric columns only (geometry is excluded automatically).
    numeric_cols = df.select_dtypes(include='number')

    # Compute summary statistics for numeric columns.
    if not numeric_cols.empty:
        summary_stats = numeric_cols.agg(['min', 'max', 'mean']).T.round(2)
        summary_df = summary_df.join(summary_stats)

    return summary_df

#______________________________________________________________________________

#______________________________________________________________________________

def group_and_sum(df, column):
    """
    Groups a DataFrame/GeoDataFrame by a specified column,
    and sums the numeric columns.
    Args:
        df: a pandas DataFrame or geopandas GeoDataFrame
        column: Column name to group by (string)
    Returns:
        pandas DataFrame with grouped data and summed numeric values
    Raises:
        ValueError: If df contains no numeric columns to sum
    """

    # Check for numeric columns before grouping.
    numeric_cols = df.select_dtypes(include='number').columns
    numeric_cols = [col for col in numeric_cols if col != column]

    if len(numeric_cols) == 0:
        raise ValueError("DataFrame contains no numeric columns to sum")

    return df.groupby(column).sum()

#______________________________________________________________________________

#______________________________________________________________________________

def min_max(df, column):
    """
    Computes the minimum and maximum values of a column.
    Args:
        df: a pandas DataFrame
        column: Column name to compute min/max for (string)
    Returns:
        (min_value, max_value) for the specified column (tuple)
    Raises:
        KeyError: If the column does not exist in the DataFrame.
    """
    return df[column].min(), df[column].max()
    
#______________________________________________________________________________

#______________________________________________________________________________

def df_to_geodataframe(df, lon_col, lat_col, crs = "EPSG:4326"):
    """
    Convert a pandas DataFrame with lon/lat columns into a GeoDataFrame.
    Ensures longitude and latitude columns to numeric values, drops
    rows with missing coordinates, and creates Point geometry objects.
    Args:
        df: pandas DataFrame
        lon_col: Longitude column name (string)
        lat_col: Latitude column name (string)
        crs: Coordinate reference system (string, default: "EPSG:4326")
    Returns:
        geopandas.GeoDataFrame with Points in specified CRS
    """

    df = df.copy()

    # Ensure latitude and longitude columns are numeric.
    df[lat_col] = pd.to_numeric(df[lat_col], errors = "coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors = "coerce")

    # Drop rows with missing data.
    df = df.dropna(subset = [lon_col, lat_col])

    # Create geometry object using Shapely Point.
    df["geometry"] = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]

    # Convert DataFrame to a GeoDataFrame.
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)

    return gdf

#______________________________________________________________________________

#______________________________________________________________________________

def create_map_orig(gdf, fields = None, title = None, color_code = None):
    """
    Create an interactive folium map.
    Centers the map of the gdf's center and handles coordinate 
    transformations. Optionally color-codes features by a specified column.
    Args:
        gdf: GeoDataFrame to visualize
        fields: field names to display in the tooltip (list, default = None)
        title: Name of map (string, default = None)
        color_code: Column name for color coding features (string, 
        default = None)
    Returns:
        folium.Map object with the GeoDataFrame layer added
    """

    # Make a copy to avoid modifying the original GeoDataFrame.
    gdf = gdf.copy()
    
    # Convert datetime columns to strings for JSON serialization.
    datetime_cols = gdf.select_dtypes(
        include = ['datetime64', 'datetime']
        ).columns
    for col in datetime_cols:
        gdf[col] = gdf[col].astype(str)

    # Calculate center based on CRS.
    # If already in geographic crs, then
    # project crs temporarily for centroid calculation.
    if gdf.crs and gdf.crs.is_geographic:
        # Use the UTM zone 17N for Detroit area.
        gdf_projected = gdf.to_crs(epsg = 32617)
        # Calculate center and project back to geographic.
        center = gdf_projected.geometry.centroid.to_crs(epsg = 4326)
    else:
        center = gdf.geometry.centroid.to_crs(epsg = 4326)

    # Calculate center
    lat = center.y.mean()
    lon = center.x.mean()

    # Convert the crs to latitude/longitude for folium mapping.
    gdf = gdf.to_crs(epsg = 4326)

    # Create a centered base map with zoom level of 11.
    map = folium.Map(
        location = [lat, lon],
        zoom_start = 11
        )

    # Configure styling based on color_code parameter.
    if color_code:
        # Get unique values and generate colors dynamically.
        unique_values = gdf[color_code].unique()
        num_colors = len(unique_values)
        # Choose colormap based on number of categories.
        if num_colors <= 10:
            cmap = plt.cm.get_cmap('tab10')
        elif num_colors <= 20:
            cmap = plt.cm.get_cmap('tab20')
        else:
            cmap = plt.cm.get_cmap('hsv')
        # Generate colors.
        colors = [mcolors.rgb2hex(
            cmap(i / num_colors)
            ) 
            for i in range(num_colors)
            ]
        # Create color map dictionary.
        color_map = {val: colors[i] for i, val in enumerate(unique_values)}
        # Define style function with color coding.
        style_function = lambda x: {
            "fillColor": color_map.get(x['properties'][color_code], 'gray'),
            "color": "black",
            "weight": 0.5,
            "fillOpacity": 0.7
        }

    # If color code is not specified, use one color.  
    else:
        # Define style function with single color.
        style_function = lambda x: {
            "fillColor": "blue",
            "color": "black",
            "weight": 0.5,
            "fillOpacity": 0.5
        }
    
    # Add GeoJSON layer to map.
    folium.GeoJson(
        gdf,
        name = title,
        style_function = style_function,
        tooltip = folium.GeoJsonTooltip(
            fields = fields
            ) 
            if fields 
            else None
    ).add_to(map)
    
    return map

#______________________________________________________________________________

#______________________________________________________________________________

def create_map(gdf, fields = None, title = None, color_code = None):
    """
    Create an interactive folium map from a GeoDataFrame.
    Centers the map on the GeoDataFrame's centroid and uses a muted 
    cmocean color scheme for better visibility.
    Args:
        gdf: geopandas.GeoDataFrame to visualize
        fields: Field names to display in tooltip (list, default = None)
        title: Name of the map layer (string, default = None)
        color_code: column name for color coding features 
        (string, default = None)
    Returns:
        folium.Map object with the GeoDataFrame layer added
    """

    gdf = gdf.copy()
    
    # Convert datetime columns to strings for JSON serialization
    datetime_cols = gdf.select_dtypes(
        include = ['datetime64', 'datetime']
        ).columns
    for col in datetime_cols:
        gdf[col] = gdf[col].astype(str)

    # Calculate center based on CRS
    # If already in geographic CRS, project temporarily for accurate centroid
    if gdf.crs and gdf.crs.is_geographic:
        # Use UTM zone 17N for Detroit area
        gdf_projected = gdf.to_crs(epsg = 32617)
        # Project back to EPSG:4326 immediately
        center = gdf_projected.geometry.centroid.to_crs(epsg = 4326)
    else:
        center = gdf.geometry.centroid.to_crs(epsg = 4326)

    # Calculate center coordinates
    lat = center.y.mean()
    lon = center.x.mean()

    # convert to WGS84 for mapping
    gdf = gdf.to_crs(epsg = 4326)

    # Create base map
    m = folium.Map(location = [lat, lon], zoom_start = 11)

    # Configure color.
    # Configure styling based on color_code parameter
    if color_code:
        unique_values = gdf[color_code].unique()
        num_colors = len(unique_values)

        # Use cmocean "phase" colormap.
        cmap = cmo.phase

        # Generate colors for discrete categories.
        colors = [mcolors.rgb2hex(
            cmap(i / max(num_colors-1, 1))
            )
            for i in range(num_colors)
            ]
        color_map = {val: colors[i] for i, val in enumerate(unique_values)}

        # Define style function with color coding.
        style_function = lambda x: {
            "fillColor": color_map.get(x['properties'][color_code], 'gray'),
            "color": "gray",
            "weight": 0.5,
            "fillOpacity": 0.35
        }

    else:
        # Define style function with single color
        style_function = lambda x: {
            "fillColor": "blue",
            "color": "black",
            "weight": 0.5,
            "fillOpacity": 0.35
        }
    
    # Add GeoJSON layer to map.
    folium.GeoJson(
        gdf,
        name = title,
        style_function = style_function,
        tooltip = folium.GeoJsonTooltip(fields = fields) if fields else None
    ).add_to(m)
    
    return m

#______________________________________________________________________________

#______________________________________________________________________________
def points_in_neighborhood_map(
    point_layers,
    neighborhoods_gdf,
    council_col = "council_district",
    neighborhood_col = "neighborhood_name",
    point_info_col = None,
    point_base_color = "blue",
    ranking_col = "med_underserv_ranking",
    start_zoom = 12,
    point_radius = 4,
    point_opacity= 0.8,
    title = None,
    subtitle = None
):
    """
    Create an interactive folium map that displays neighborhood polygons
    color-coded by council-district, and overlays point layers showing 
    contextual information including council_district, neighborhood, etc.
    Each point's shade corresponds to the ranking column's value.
    
    Args:
        point_layers: Dictionary of point layers to overlay 
            {'name': GeoDataFrame}
        neighborhoods_gdf: GeoDataFrame containing Neighborhood polygons 
            and council district column (GeoDataFrame)
        council_col: name of council district column (string)
        neighborhood_col: Name of neighborhood column (string)
        point_info_col: Column in point layer to display in tooltip (string, 
            optional)
        point_base_color: Base color for points (string)
        ranking_col: Column to dertermine point shade where lower values 
            are darker, higher values are lighter (string)
        start_zoom: Initial zoom level (integer)
        point_radius: Radius of point markers (integer)
        point_opacity: Opacity of point markers (0-1) (float)
        title: Map title displayed in top-left corner (string)
        subtitle: Subtitle displayed below title (string)
    
    Returns:
        folium.Map
    """

    # Normalize crs and ensure all geometries use WGS84 for web mapping
    neighborhoods = neighborhoods_gdf.to_crs(epsg = 4326)
    normalized_points = {}
    all_points = []

    # Reproject each point layer
    for name, gdf in point_layers.items():
        gdf = gdf.to_crs(epsg = 4326)
        normalized_points[name] = gdf
        all_points.append(gdf)

    # Initialize the map
    combined_points = gpd.GeoDataFrame(
        pd.concat(all_points, ignore_index = True), 
        crs = "EPSG:4326"
    )
    # Calculate centroid of all points
    center_lat = combined_points.geometry.y.mean()
    center_lon = combined_points.geometry.x.mean()
    m = folium.Map(
        location = [center_lat, center_lon], 
        zoom_start = start_zoom
    )

    # Add title block to top left if provided
    if title:
        title_html = f"""
        <div style="
            position: fixed;
            top: 15px;
            left: 15px;
            z-index: 9999;
            background-color: white;
            padding: 8px 12px;
            border-radius: 6px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.2);
            font-family: Arial, sans-serif;
        ">
            <div style="font-size: 14px; font-weight: bold;">
                {title}
            </div>
            {f'<div style="font-size: 12px; color: gray;">{subtitle}</div>' 
             if subtitle else ''}
        </div>
        """
        m.get_root().html.add_child(folium.Element(title_html))

    # Generate distinct colors to color code each council district
    unique_vals = neighborhoods[council_col].dropna().unique()
    num_colors = len(unique_vals)
    cmap = cmo.phase
    colors = [
        mcolors.rgb2hex(cmap(i / max(num_colors - 1, 1))) 
        for i in range(num_colors)
    ]
    color_map = {val: colors[i] for i, val in enumerate(unique_vals)}

    # Add neighborhood polygons as GeoJSON layer
    folium.GeoJson(
        neighborhoods,
        name="Neighborhoods",
        style_function = lambda feature: {
            "fillColor": color_map.get(
                feature["properties"].get(council_col), "gray"
            ),
            "color": "black",
            "weight": 0.5,
            "fillOpacity": 0.35,
        },
        tooltip = folium.GeoJsonTooltip(
            fields = [council_col, neighborhood_col],
            aliases = ["Council District:", "Neighborhood:"],
        ),
    ).add_to(m)

    # Set up ranking-based color gradient
    base_rgb = mcolors.to_rgb(point_base_color)

    # Determine global min/max rankings for consistent shading
    ranking_exists = any(
        ranking_col in gdf.columns for gdf in normalized_points.values()
    )
    if ranking_exists:
        all_ranks = pd.concat([
            gdf[ranking_col] for gdf in normalized_points.values() 
            if ranking_col in gdf.columns
        ])
        rank_min = all_ranks.min()
        rank_max = all_ranks.max()

    # Add each point layer through spatial join with neighborhoods
    for layer_name, points_gdf in normalized_points.items():
        # Spacial join to get neighborhood attributes
        joined = gpd.sjoin(
            points_gdf,
            neighborhoods[[neighborhood_col, council_col, "geometry"]],
            how = "left",
            predicate = "within"
        )

        # Add individual point markers
        for _, row in joined.iterrows():
            # Create tooltip content
            tooltip_text = (
                f"Neighborhood: {row.get(neighborhood_col, 'N/A')}<br>"
                f"Council District: {row.get(council_col, 'N/A')}<br>"
            )
            if point_info_col:
                tooltip_text += (
                    f"{point_info_col}: {row.get(point_info_col, 'N/A')}"
                )

            # Calculate point shade based on ranking column
            if (ranking_exists and ranking_col in row and 
                    pd.notnull(row.get(ranking_col))):
                # Normalize ranking to range from 0 to 1
                rank_range = max(rank_max - rank_min, 1e-6)
                norm_rank = (row[ranking_col] - rank_min) / rank_range
                # Lower ranking is darker, higher is lighter
                color_rgb = tuple([
                    base_rgb[i] * (1 - 0.5 * norm_rank) 
                    for i in range(3)
                ])
                fill_color = mcolors.to_hex(color_rgb)
            else:
                fill_color = point_base_color

            folium.CircleMarker(
                location = [row.geometry.y, row.geometry.x],
                radius = point_radius,
                color = fill_color,
                fill = True,
                fill_color = fill_color,
                fill_opacity = point_opacity,
                tooltip = tooltip_text,
            ).add_to(m)

    return m

#______________________________________________________________________________

#______________________________________________________________________________

def assign_points_to_neighborhood(
    points_gdf, neighborhoods_gdf, neighborhood_cols):
    """
    Performs a spatial join to determine which neighborhood polygon each point
    falls into, then attaches neighborhood information to each point.

    Args:
        points_gdf: Point locations (GeoDataFrame)
        neighborhoods_gdf: Neighborhood polygons (GeoDataFrame)
        neighborhood_cols: Columns from neighborhoods_gdf to attach (list[str])

    Returns: 
        GeoDataFrame: a copy of points_gdf with additional columns from 
        neighborhood_cols. Retains all original points (left join).
    """

    # Make a copy to avoid modifying original.
    points = points_gdf.copy()

    # Automatically set crs to points.crs if they differ.
    if neighborhoods_gdf.crs != points.crs:
        neighborhoods_gdf = neighborhoods_gdf.to_crs(points.crs)

    # Left join neighborhoods to points
    joined = gpd.sjoin(
        points,
        neighborhoods_gdf[['geometry'] + neighborhood_cols],
        how = 'left',
        predicate = 'within'
    )

    # drop spatial join helper column
    joined = joined.drop(columns = 'index_right')

    return joined

#______________________________________________________________________________

#______________________________________________________________________________
def density_per_neighborhood(
    points_gdf, 
    neighborhoods_gdf,
    neighborhood_col = 'Current_Neighborhoods',
    neighborhood_name_col = 'neighborhood_name',
    council_col = 'council_district'):
    """
    Calculate points per square mile for neighborhoods and attach council 
    district, including raw point counts.

    Returns a DataFrame with columns:
    [neighborhood_col, points, points_per_sq_mile, council_col]
    """

    # Count points per neighborhood (raw counts)
    counts = (
        points_gdf
        .groupby(neighborhood_col)
        .size()
        .rename('points')  # store raw counts
        .reset_index()
    )

    # Calculate neighborhood areas in square miles (crs in feet)
    neighborhoods_gdf = neighborhoods_gdf.to_crs(epsg = 2263) 
    neighborhoods_gdf['area_sq_miles'] = (
        neighborhoods_gdf.geometry.area / (5280 ** 2)
    )

    # Merge in council district and area 
    merged = counts.merge(
        neighborhoods_gdf[
            [neighborhood_name_col, council_col, 'area_sq_miles']
        ],
        left_on = neighborhood_col,
        right_on = neighborhood_name_col,
        how = 'left'
    )

    # Calculate points per square mile and round to nearest tenth
    merged['points_per_sq_mile'] = (
        (merged['points'] / merged['area_sq_miles']).round(1)
        )

    # Keep desired columns
    merged = merged[
        [neighborhood_col, 'points', 'points_per_sq_mile', council_col]
    ]

    # Sort by density descending
    merged = merged.sort_values(
        'points_per_sq_mile', 
        ascending = False
        ).reset_index(drop = True)

    return merged

#______________________________________________________________________________


#______________________________________________________________________________

def flag_underserved_with_green_lights(
    med_counts, green_counts, threshold = 5):
    """
    Identify neighborhoods that are medically underserved and
    have more than or equal to the green light location threshold.

    Args:
        med_counts: 
            Count of MUAP points in neighborhood (DataFrame)
        green_counts: 
            Count of Green Light locations in neighborhood (DataFrame)
        threshold: 
            Minimum number of Green Light Locations for flag

    Returns:
        merged: DataFrame with medical counts, green light counts, 
            and a flag for neighborhoods that meet threshold
    """

    # Join the DataFrames
    merged = med_counts.merge(
        green_counts,
        on = 'City Neighborhoods', 
        how = 'left', 
        suffixes = ('_med', '_green')
    )

    # Fill NaN values with 0
    merged['count_green'] = merged['count_green'].fillna(0)

    # Creates boolean column (True if >= 5)
    merged['flag_many_green'] = merged['count_green'] >= threshold

    return merged

#______________________________________________________________________________


#______________________________________________________________________________

def neighborhood_choropleth_map(
    neighborhoods_gdf,
    value_col = "request_count",
    council_col = "council_district",
    neighborhood_col = "neighborhood_name",
    start_zoom = 12,
    title = None,
    subtitle = None):
    """
    Display a choropleth map of neighborhoods colored by council district,
    with shading based on request count and hover tooltips.
    Optional fixed title and subtitle in the top-left corner.

    Args:
        neighborhoods_gdf: Neighborhood polygons with request counts 
            (GeoDataFrame)
        value_col: Column with aggregated request counts (string)
        council_col: Council district column (str)
        neighborhood_col: Neighborhood name column (str)
        start_zoom: Initial zoom level (int)
        title: Optional map title (str)
        subtitle: Optional map subtitle (str)

    Returns:
        folium.Map
    """

    # Normalize the coordinate reference system
    neighborhoods = neighborhoods_gdf.to_crs(epsg = 4326)

    # Fill missing values with 0
    # This allows us to include neighborhoods with no requests in map
    neighborhoods[value_col] = neighborhoods[value_col].fillna(0)

    # Calculate the map center
    center_lat = neighborhoods.geometry.centroid.y.mean()
    center_lon = neighborhoods.geometry.centroid.x.mean()
    m = folium.Map(
        location = [center_lat, center_lon], zoom_start = start_zoom
    )

    # Add title/subtitle HTML overlay in top-left 
    if title or subtitle:
        html_content = ""
        if title:
            html_content += f"<h4 style='margin:0'>{title}</h4>"
        if subtitle:
            html_content += f"<p style='margin:0'>{subtitle}</p>"

        # CSS styling for fixed position overlay with semi-transparent background
        template = f"""
        <div style="
            position: fixed;
            top: 10px;
            left: 10px;
            z-index: 9999;
            background-color: rgba(255,255,255,0.85);
            padding: 5px 10px;
            border-radius: 5px;
            font-family: sans-serif;
        ">
            {html_content}
        </div>
        """
        # Add the HTML template to the map
        macro = MacroElement()
        macro._template = Template(template)
        m.get_root().add_child(macro)

    # Create council district base colors
    # Get all unique council districts
    unique_vals = neighborhoods[council_col].dropna().unique()
    # Align number of colors to number of districts
    num_colors = len(unique_vals)
    # Use the phase colormap
    cmap = cmo.phase
    # Generate evenly-spaced colors from colormap
    # This ensures distinct coloring
    colors = [
        mcolors.rgb2hex(cmap(i / max(num_colors - 1, 1)))
        for i in range(num_colors)
    ]
    # Create a map for council district to color
    council_color_map = {val: colors[i] for i, val in enumerate(unique_vals)}

    # Normalize request counts for shading
    # Find the range of request counts
    vmin = neighborhoods[value_col].min()
    vmax = neighborhoods[value_col].max()

    def style_function(feature):
        """
        Define style for each polygon.
        Combines council distict color with request count shade.
        """

        # Extract properties from the GeoJSON feature
        props = feature["properties"]
        council = props.get(council_col)
        count = props.get(value_col)

        # Get the base color for the council district
        base_color = council_color_map.get(council, "#cccccc")

        # Apply shading based on request count
        if count is not None and vmax > vmin:
            # Normalize the count to 0 to 1
            norm = (count - vmin) / (vmax - vmin)
            # Convert hex to RGB to manipulate
            base_rgb = mcolors.to_rgb(base_color)
            # Darken color based on request count
            shade_rgb = tuple(c * (0.4 + 0.6 * (1 - norm)) for c in base_rgb)
            # Convert back to hex
            fill_color = mcolors.to_hex(shade_rgb)
        else:
            # Use base color if there is no count
            fill_color = base_color

        # Return the style dictionary for the polygon
        return {
            "fillColor": fill_color,
            "color": "black",
            "weight": 0.5,
            "fillOpacity": 0.6,
        }

    # GeoJson with hover
    # Add neighborhood polygons to the map
    folium.GeoJson(
        neighborhoods,
        name = "Neighborhood Requests",
        style_function = style_function,
        # Configure the tooltip
        tooltip = folium.GeoJsonTooltip(
            fields = [council_col, neighborhood_col, value_col],
            aliases = [
                "Council District:",
                "Neighborhood:",
                "Request Count:",
            ],
            localize = True,
            sticky = True,
        ),
    ).add_to(m)

    return m
#______________________________________________________________________________


#______________________________________________________________________________

# Calculate the average points_per_square_mile 
# and the min/max points_per_square_mile per council district

def council_district_stats(
    neighborhood_density_df, 
    council_col = None, 
    neighborhood_col = None,  
    density_col = 'points_per_sq_mile',
    points_col = 'points'):
    """
    Calculate higher-level district statistics of points per square mile
    (council districts, neighborhood clusters, etc.).

    Args:
        neighborhood_density_df: Neighborhood-level density with columns for 
            neighborhood, density, and higher-level grouping. (pd.DataFrame)
        council_col: Column name for higher-level grouping 
            (e.g., council_district, neighborhood_cluster) (string)
        neighborhood_col: Column name for neighborhood in the DataFrame 
            (string)
        density_col: Column name for points per square mile (string)
        points_col: Column name for total points in the neighborhood (string)

    Returns:
        District-level stats: average, min, max density and neighborhoods 
        corresponding to min/max (pd.DataFrame)
    """

#______________________________________________________________________________

    # Validate input columns exist
    for col in [council_col, neighborhood_col, density_col, points_col]:
        if col not in neighborhood_density_df.columns:
            available = neighborhood_density_df.columns.tolist()
            raise ValueError(
                f"Column '{col}' not found in DataFrame. "
                f"Available columns: {available}"
            )
    # Group by the higher-level district and calculate mean/min/max densities
    council_stats = (
        neighborhood_density_df
        .groupby(council_col)[density_col]
        .agg(
            avg_points_per_sq_mile='mean',
            min_points_per_sq_mile='min',
            max_points_per_sq_mile='max'
        )
        .round(1)
        .reset_index()
    )

    # Total points per district
    total_points = (
        neighborhood_density_df
        .groupby(council_col)[points_col]
        .sum()
        .reset_index(name = 'total_points')
    )

    council_stats = council_stats.merge(total_points, on = council_col)

    # Find neighborhoods corresponding to min and max densities
    min_neighborhoods = (
        neighborhood_density_df
        .groupby(council_col)
        .apply(
            lambda df: ', '.join(
                df.loc[
                    df[density_col] == df[density_col].min(),
                    neighborhood_col
                ]
            )
        )
        .reset_index(name = 'min_neighborhood')
    )

    max_neighborhoods = (
        neighborhood_density_df
        .groupby(council_col)
        .apply(
            lambda df: ', '.join(
                df.loc[
                    df[density_col] == df[density_col].max(),
                    neighborhood_col
                ]
            )
        )
        .reset_index(name = 'max_neighborhood')
    )

    council_stats = council_stats.merge(
        min_neighborhoods, 
        on = council_col
    )
    council_stats = council_stats.merge(
        max_neighborhoods, 
        on = council_col
    )

    # Sort by average density descending
    council_stats = council_stats.sort_values(
        'avg_points_per_sq_mile',
        ascending = False
    ).reset_index(drop = True)

    return council_stats
#______________________________________________________________________________

#______________________________________________________________________________

def calculate_district_imu(
    current_neighborhoods_gdf,
    medically_underserved_gdf,
    district_col = "council_district",
    neighborhood_col = "neighborhood_name",
    imu_col = "med_underserv_ranking",
    non_residential_names = None,
    fallback_value = 70):
    """
    Calculate area-weighted Index of Medical Underservice (IMU) for each
    district and identify the most underserved neighborhood per district.

    Args:
        current_neighborhoods_gdf: GeoDataFrame of neighborhood polygons
        medically_underserved_gdf: GeoDataFrame of IMU areas/rankings
        district_col: Column name for district identifier (string)
        neighborhood_col: Column name for neighborhood name (string)
        imu_col: Column name for medical underservice ranking (string)
        non_residential_names: Set of neighborhood names to exclude (set)
        fallback_value: IMU value for areas with no data (int/float)

    Returns:
        DataFrame with district-level IMU statistics
    """

    # Filter out non-residential neighborhoods
    if non_residential_names is not None:
        current_neighborhoods_gdf = current_neighborhoods_gdf[
            ~current_neighborhoods_gdf[neighborhood_col].isin(
                non_residential_names
            )
        ].copy()

    # Project to equal-area crs
    neighborhoods_proj = current_neighborhoods_gdf.to_crs(epsg = 3857)
    imu_proj = medically_underserved_gdf.to_crs(epsg = 3857)

    # Spatial join to match neighborhoods with MUAPs
    joined = gpd.sjoin(
        neighborhoods_proj,
        imu_proj,
        how = "left",
        predicate  ="intersects"
    )

    # Aggregate IMU at neighborhood level 
    # Take minimum if there's multiple overlaps
    neighborhood_imu = (
        joined.groupby([neighborhood_col, district_col, "geometry"])
        [imu_col]
        .min()
        .reset_index()
    )

    # Make it a GeoDataFrame so we can compute area
    neighborhood_imu = gpd.GeoDataFrame(
        neighborhood_imu, 
        geometry = "geometry", 
        crs = current_neighborhoods_gdf.crs
    )

    # Assign fallback IMU t neighborhoods missing data
    neighborhood_imu[imu_col] = neighborhood_imu[imu_col].fillna(
        fallback_value
    )

    # Compute area in square miles (1 sq mi = 2,589,988.11 sq meters)
    neighborhood_imu["area_sq_mi"] = (
        neighborhood_imu.geometry.area / 2_589_988.11
    )

    # Calculate weighted IMU (IMU score × area)
    neighborhood_imu["weighted_imu"] = (
        neighborhood_imu[imu_col] * neighborhood_imu["area_sq_mi"]
    )

    # Aggregate by district
    district_agg = (
        neighborhood_imu
        .groupby(district_col)
        .agg(
            total_weighted_imu = ("weighted_imu", "sum"),
            total_area = ("area_sq_mi", "sum")
        )
        .reset_index()
    )

    # Calculate area-weighted average IMU per district
    district_agg["area_weighted_IMU"] = (
        district_agg["total_weighted_imu"] / district_agg["total_area"]
    ).round(0).astype(int)

    # Find the most underserved neighborhood in each district (lowest IMU)
    min_neighborhood = (
        neighborhood_imu.groupby(district_col)
        .apply(
            lambda x: x.loc[
                x[imu_col].idxmin(),
                [neighborhood_col, imu_col]
            ]
        )
        .reset_index()
        .rename(columns = {
            neighborhood_col: "lowest_ranked_neighborhood",
            imu_col: "lowest_neighborhood_IMU"
        })
    )

    # Merge district-level stats with most underserved neighborhoods
    district_results = district_agg.merge(
        min_neighborhood,
        on = district_col
    )

    # Sort by area-weighted IMU (ascending = most underserved first)
    district_results = district_results.sort_values(
        "area_weighted_IMU"
    ).reset_index(drop = True)

    return district_results[
        [
            district_col,
            "area_weighted_IMU",
            "lowest_ranked_neighborhood",
            "lowest_neighborhood_IMU"
        ]
    ]

#______________________________________________________________________________
