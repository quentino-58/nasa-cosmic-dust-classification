import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import csv
import os

import jax
from jax import numpy as jnp
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import optax

def opening(global_catalog_path : str, folder_path : str, removing_copy=False) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Open all the files in a given folder (with the csv files of the spectra). Also build a vector with the particle type according to NASA classification.
    
    Parameters:
        global_catalog_path (str): The path to the csv file containing all the particles information.
        folder_path (str): The path to the folder.
        removing_copy (bool): If TRUE remove the copy in the input data.

    Returns:
        spectra_list (np.ndarray): An array containing the vectors of the spectra. 
        name_list (np.ndarray): An array containing the names of the particles. 
        type_list (np.ndarray): An array containing the types of the particle. 
        certainty_list (np.ndarray): An array to know if the type is certain according to NASA. 
        sizes_list (np.ndarray): An array containing all the sizes of the particle.
        shape_list (np.ndarray): An array containing all the shapes of the particle.
        color_list (np.ndarray): An array containing all the colors of the particle.
    """     

    # Opening the global catalog
    with open(global_catalog_path, 'r') as csv_global:
        reader = csv.reader(csv_global)
        all_data = np.array([row for row in reader], dtype=str)[1:]

        particle_names = all_data[:,0]
        types_uncertain = all_data[:,2]
        types_certain = all_data[:,3]
        sizes = all_data[:,4]
        colors = all_data[:,5]
        shapes = all_data[:,6]
        years = all_data[:,7]

    spectra_list = []
    name_list = []
    type_list = []
    certainty_list = []
    years_list = []
    sizes_list = []
    shapes_list = []
    colors_list = []

    # Reading the file
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = folder_path + "/" + file
            particle_name = file[:-4].replace("_","")

            # Opening the particle vector
            with open(file_path, 'r') as csv_file:
                reader = csv.reader(csv_file)
                current_vector = [row[1] for row in reader][1:]

            # Getting the line corresponding to the particle
            idx_line = np.where(particle_names == particle_name)[0]

            if len(idx_line) == 0:
                print(f"No match in global catalog: {particle_name}")
            elif len(idx_line) > 1:
                print(f"Multiple matches in global catalog: {particle_name}")
                print(all_data[idx_line])

            # Getting the type
            particle_type = types_certain[idx_line]

            # Getting the certainty of the type
            certainty = types_uncertain[idx_line]

            # Getting the size
            size = sizes[idx_line]

            # Getting the year
            year = years[idx_line]

            # Getting the shape
            shape = shapes[idx_line]

            # Getting the color
            color = colors[idx_line]

            # Filling the results in the data
            spectra_list.append(np.array(current_vector, dtype=np.float64))
            name_list.append(particle_name)
            type_list.append(particle_type)
            shapes_list.append(shape)
            colors_list.append(color)
            certainty_list.append(certainty)
            sizes_list.append(size)
            years_list.append(year)


    # Removing copy if needed
    if removing_copy == True:
        spectra_list, indices = np.unique(spectra_list, axis=0, return_index=True)
        if name_list is not None:
            name_list = np.array(name_list)[indices]
        if type_list is not None:
            type_list = np.array(type_list)[indices]
        if certainty_list is not None:
            certainty_list = np.array(certainty_list)[indices]
        if sizes_list is not None:
            sizes_list = np.array(sizes_list)[indices]
        if years_list is not None:
            years_list = np.array(years_list)[indices]
        if shapes_list is not None:
            shapes_list = np.array(shapes_list)[indices]
        if colors_list is not None:
            colors_list = np.array(colors_list)[indices]

    

    spectra_list = np.array(spectra_list)
    name_list = np.array(name_list)
    type_list = np.array(type_list)
    certainty_list = np.array(certainty_list)
    sizes_list = np.array(sizes_list, dtype=float)
    years_list = np.array(years_list, dtype=int)
    shapes_list = np.array(shapes_list)
    colors_list = np.array(colors_list)

    return spectra_list, name_list, type_list, certainty_list, sizes_list, years_list, shapes_list, colors_list


def NASA_color(type_list : np.ndarray, certainty_list : np.ndarray, dimensions : int, using_plotly : bool) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Building vectors used to plot the result, using NASA symbols and colors.  

    Parameters:
        type_list (np.ndarray): An array containing the types of the particle. 
        certainty_list (np.ndarray): An array to know if the type is certain according to NASA.
        dimensions (int): The number of dimensions for the plot.
        using_plotly (bool): True or False.

    Returns:
        colors (np.ndarray): An array containing the color of the particle. 
        markers (np.ndarray): An array to know if the marker of the particle. 
        legend_elements (list): The legend of the graph.
    """        

    particle_mapping = {'C' + 'Y': ('red', 'o'),
                        'TCA' + 'Y': ('green', 's'),
                        'TCN' + 'Y': ('blue', '^'),
                        'AOS' + 'Y': ('purple', 'x'),
                        
                        'C' + 'N': ('gray', 'o'),
                        'TCA' + 'N': ('gray', 's'),
                        'TCN' + 'N': ('gray', '^'),
                        'AOS' + 'N': ('gray', 'x'),
                        
                        'NA' + 'NA': ('black', '+')}
    
    changing = {
        'o': 'circle',
        's': 'square',
        '^': 'diamond',
        'x': 'cross',
        '+': 'x'
    }

    if dimensions == 2:
        particle_color, particle_marker = zip(*(particle_mapping.get(str(type[0]) + str(certainty[0]), ('black', '+')) for type, certainty in zip(type_list, certainty_list)))

    elif dimensions == 3:
        particle_color, particle_marker = zip(*(particle_mapping.get(str(type[0]) + str(certainty[0]), ('black', '+')) for type, certainty in zip(type_list, certainty_list)))
    
    if using_plotly == True:
        particle_marker = [changing[m] if m in changing else m for m in particle_marker]

    # Building the legend
    legend_elements = [mpatches.Patch(color=color, label=label) for color, label in zip(['red', 'blue', 'green', 'purple', 'gray'],
                                                                                        ['C', 'TCN', 'TCA', 'AOS', 'Uncertain'])]
    return particle_color, particle_marker, legend_elements


def clustering_color(clusters: np.ndarray, nmb_clusters: int, name_clusters: list):
    """Return particle colors, markers and legend elements for clustering plots."""

    colors = [
        (0.1216,0.4667,0.7059,1),(1,0.498,0.0549,1),(0.1725,0.6275,0.1725,1),
        (0.8392,0.1529,0.1569,1),(0.5804,0.4039,0.7412,1),(0.549,0.3373,0.2941,1),
        (0.8902,0.4667,0.7608,1),(0.498,0.498,0.498,1),(0.7373,0.7412,0.1333,1),
        (0.0902,0.7451,0.8118,1)
    ]

    markers = ['o', 's', '^', 'X']

    def style(i):
        if 0 <= i < nmb_clusters:
            return colors[i % len(colors)], markers[i % len(markers)]
        return 'black', '+'

    particle_color, particle_marker = zip(*(style(i) for i in clusters))

    legend_elements = [
        mlines.Line2D([0],[0],
            marker=markers[i % len(markers)],
            color='none',
            markerfacecolor=colors[i % len(colors)],
            linestyle='None',
            label=name_clusters[i]
        )
        for i in range(nmb_clusters)
    ]

    if np.any((clusters < 0) | (clusters >= nmb_clusters)):
        legend_elements.append(
            mlines.Line2D([0],[0], marker='+', color='black',
                          linestyle='None', label='Unknown')
        )

    return np.array(particle_color), np.array(particle_marker), legend_elements

def clustering_color_old(clusters: np.ndarray, nmb_clusters: int, name_clusters: list) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Building vectors used to plot the result, using the clusters calculated by the algorithm.  

    Parameters:
        clusters (np.ndarray): An array containing the cluster numbers for each particle. 
        nmb_clusters (int): The number of clusters. 
        name_clusters (list): The names of the clusters (ordered by cluster index).

    Returns:
        particle_color (np.ndarray): Color of each particle. 
        particle_marker (np.ndarray): Marker of each particle. 
        legend_elements (list): Legend elements for plotting.
    """

    colors = [
        (0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0),
        (1.0, 0.4980392156862745, 0.054901960784313725, 1.0),
        (0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0),
        (0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0),
        (0.5803921568627451, 0.403921568627451, 0.7411764705882353, 1.0),
        (0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0),
        (0.8901960784313725, 0.4666666666666667, 0.7607843137254902, 1.0),
        (0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1.0),
        (0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1.0),
        (0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0),
        (0.6823529411764706, 0.7803921568627451, 0.9098039215686274, 1.0),
        (1.0, 0.7333333333333333, 0.47058823529411764, 1.0),
        (0.596078431372549, 0.8745098039215686, 0.5411764705882353, 1.0),
        (1.0, 0.596078431372549, 0.5882352941176471, 1.0),
        (0.7725490196078432, 0.6901960784313725, 0.8352941176470589, 1.0),
        (0.7686274509803922, 0.611764705882353, 0.5803921568627451, 1.0),
        (0.9686274509803922, 0.7137254901960784, 0.8235294117647058, 1.0),
        (0.7803921568627451, 0.7803921568627451, 0.7803921568627451, 1.0),
        (0.8588235294117647, 0.8588235294117647, 0.5529411764705883, 1.0),
        (0.6196078431372549, 0.8549019607843137, 0.8980392156862745, 1.0),
        (0.8549019607843137, 0.6470588235294118, 0.12549019607843137, 1.0),
        (0.5, 0.5, 0.5, 1.0),
        (0.6, 0.4, 0.8, 1.0),
        (0.8, 0.4, 0.4, 1.0)
    ]

    markers = ['o', 's', '^', 'X']

    # Map cluster index -> (color, marker, name)
    particle_mapping = {
        i: (colors[i], markers[i % len(markers)], name_clusters[i])
        for i in range(nmb_clusters)
    }

    # Assign color and marker to each particle
    particle_color, particle_marker = zip(
        *(particle_mapping.get(cluster_index, ('black', '+', 'Unknown'))[:2]
          for cluster_index in clusters)
    )

    # Build legend using cluster names
    legend_elements = [
        mlines.Line2D(
            [0], [0],
            marker=particle_mapping[i][1],
            linewidth=0,
            markeredgecolor='None',
            markerfacecolor=particle_mapping[i][0],
            linestyle='None',
            label=name_clusters[i]
        )
        for i in range(nmb_clusters)
    ]

    return np.array(particle_color), np.array(particle_marker), legend_elements


def mixing_color_cat(catalog_list : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Building vectors used to plot the result, using a different color for each catalog.  

    Parameters:
        catalog_list (np.ndarray): An array containing the catalog number of the particle. 

    Returns:
        colors (np.ndarray): An array containing the color of the particle. 
        markers (np.ndarray): An array to know if the marker of the particle. 
        legend_elements (list): The legend of the graph.
    """        

    particle_mapping = {'1_1': ((0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0), 'o'),
                        '1_2': ((1.0, 0.4980392156862745, 0.054901960784313725, 1.0), 's'),
                        '2_1': ((0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0), '^'),
                        '2_2': ((0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0), 'x'),
                        '3': ((0.5803921568627451, 0.403921568627451, 0.7411764705882353, 1.0), 'o'),
                        '4_1': ((0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0), 's'),
                        '4_2': ((0.8901960784313725, 0.4666666666666667, 0.7607843137254902, 1.0), '^'),
                        '5': ((0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1.0), 'x'),
                        '6': ((0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1.0), 'o'),
                        '7': ((0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0), 's'),
                        '8': ((0.6823529411764706, 0.7803921568627451, 0.9098039215686274, 1.0), '^'),
                        '9': ((1.0, 0.7333333333333333, 0.47058823529411764, 1.0), 'x'),
                        '10': ((0.596078431372549, 0.8745098039215686, 0.5411764705882353, 1.0), 'o'),
                        '11': ((1.0, 0.596078431372549, 0.5882352941176471, 1.0), 's'),
                        '12': ((0.7725490196078432, 0.6901960784313725, 0.8352941176470589, 1.0), '^'),
                        '13': ((0.7686274509803922, 0.611764705882353, 0.5803921568627451, 1.0), 'x'),
                        '14': ((0.9686274509803922, 0.7137254901960784, 0.8235294117647058, 1.0), 'o'),
                        '15': ((0.7803921568627451, 0.7803921568627451, 0.7803921568627451, 1.0), 's'),
                        '17': ((0.8588235294117647, 0.8588235294117647, 0.5529411764705883, 1.0), '^'),
                        '18': ((0.6196078431372549, 0.8549019607843137, 0.8980392156862745, 1.0), 'x'),
                        '19': ((0.403921568627451, 0.6627450980392157, 0.8117647058823529, 1.0), 'o'), 
                        '20': ((0.8549019607843137, 0.6470588235294118, 0.12549019607843137, 1.0), 's'),
                        '21': ((0.5, 0.5, 0.5, 1.0), '^'), 
                        '22': ((0.6, 0.4, 0.8, 1.0), 'x'), 
                        '23': ((0.8, 0.4, 0.4, 1.0), 'o')}


    particle_color, particle_marker = zip(*(particle_mapping.get(str(catalog_number), ('black', '+')) for catalog_number in catalog_list))

    # Building the legend
    legend_elements = [mpatches.Patch(color=color, label= "Catalog " + label) for color, label in zip([particle_mapping.get(str(catalog_index), ('black', '+'))[0] for catalog_index in np.unique(catalog_list)], 
                                                                                       [str(catalog_index) for catalog_index in np.sort(np.unique(catalog_list))])]

    return particle_color, particle_marker, legend_elements

def mixing_color_years(
    years_list: np.ndarray,
    years_to_display: list[int]
):
    """
    Assign colors and markers by year, and return a mask for visible points.
    """

    years_list = np.asarray(years_list).astype(int).ravel()

    # Visibility mask
    if len(years_to_display) == 0:
        mask = np.zeros(len(years_list), dtype=bool)
    else:
        years_to_display = np.asarray(years_to_display, dtype=int)
        mask = np.isin(years_list, years_to_display)

    unique_years = np.sort(np.unique(years_list[mask])) if mask.any() else []

    cmap = plt.get_cmap("tab20")
    markers_cycle = ['o', 's', '^', 'x', 'D', 'P', '*', 'v', '<', '>']

    year_mapping = {
        year: (cmap(i % cmap.N), markers_cycle[i % len(markers_cycle)])
        for i, year in enumerate(unique_years)
    }

    particle_color = [
        year_mapping[year][0] if visible else 'none'
        for year, visible in zip(years_list, mask)
    ]

    particle_marker = [
        year_mapping[year][1] if visible else None
        for year, visible in zip(years_list, mask)
    ]

    legend_elements = [
        mpatches.Patch(color=year_mapping[year][0], label=f"Year {year}")
        for year in unique_years
    ]

    return particle_color, particle_marker, mask, legend_elements


def mixing_color_years_1(years_list: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Building vectors used to plot the result, using a different color for each year.
    """

    years_list = np.asarray(years_list).astype(int).ravel()

    unique_years = np.sort(np.unique(years_list))

    cmap = plt.get_cmap("tab20")
    markers_cycle = ['o', 's', '^', 'x', 'D', 'P', '*', 'v', '<', '>']

    year_mapping = {}
    for i, year in enumerate(unique_years):
        color = cmap(i % cmap.N)
        marker = markers_cycle[i % len(markers_cycle)]
        year_mapping[year] = (color, marker)

    particle_color, particle_marker = zip(
        *(year_mapping.get(year, ('black', '+')) for year in years_list)
    )

    legend_elements = [
        mpatches.Patch(color=year_mapping[year][0], label=f"{year}")
        for year in unique_years
    ]

    return (
        np.array(particle_color),
        np.array(particle_marker),
        legend_elements
    )


def hex_to_RGB(hex_str : str) -> tuple[int, int, int]:
    """ #FFFFFF -> [255,255,255]"""
    # Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def fine_progression(x, base=10):
    return (1 - np.exp(-base * x)) / (1 - np.exp(-base))

def get_color_gradient(color1 : str, color2 : str, sizes_list : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Given two hex colors, returns a color gradient with n colors.

    Parameters:
        color1 (str): A string giving the color (ex : "#8A5AC2"). 
        color2 (str): A string giving the color (ex : "#8A5AC2").  
        sizes_list (np.ndarray): The array containing the sizes of the particle.

    Returns:
        An array containing all the colors following the gradient related to their sizes.
    """
    # The two limits colors
    color1_rgb = np.array(hex_to_RGB(color1))/255
    color2_rgb = np.array(hex_to_RGB(color2))/255

    # Normalizing the sizes
    sizes_list = sizes_list / (np.max(sizes_list) - np.min(sizes_list))
    
    output_colors = [((1 - fine_progression(x)) * color1_rgb + (fine_progression(x) * color2_rgb)) for x in sizes_list]
    
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in output_colors]


def color_gradient_NASA(type_list: np.ndarray, sizes_list: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Building vectors used to plot the result, using a different color for each catalog.  

    Parameters:
        name_list (np.ndarray): An array containing the names of the particle.
        type_list (np.ndarray): An array containing the types of the particle. 
        sizes_list (np.ndarray): An array containing the sizes of the particle. 

    Returns:
        colors (np.ndarray): An array containing the color of the particle. 
        markers (np.ndarray): An array containing the marker type of the particle. 
        legend_elements (list): The legend of the graph.
        legend_elements_bis (list): The legend giving the size of the particles.
        sizes (np.ndarray): The sizes of the points on the graph.
    """

    # Define color mapping
    color_mapping_facecolor = {'C': 'red',
                     'TCA': 'green',
                     'TCN': 'blue',
                     'AOS': 'purple',
                     'NA': 'black',
                     'M': 'grey',
                     'AOS?' : 'None',
                     'C?' : 'None',
                     'TCA?' : 'None',
                     'TCN?' : 'None',
                     'ELMT': 'yellow'}
    
    color_mapping_edgecolor = {'C': 'red',
                    'TCA': 'green',
                    'TCN': 'blue',
                    'AOS': 'purple',
                    'NA': 'black',
                    'M': 'black',
                    'AOS?' : 'purple',
                    'C?' : 'red',
                    'TCA?' : 'green',
                    'TCN?' : 'blue',
                    'ELMT': 'black'}

    particle_color = np.array([color_mapping_facecolor.get(type[0]) for type in type_list], dtype=str)
    particle_edgecolor = np.array([color_mapping_edgecolor.get(type[0]) for type in type_list], dtype=str)

    # Define marker mapping
    #particle_marker = np.array(['o' if t != 'M' else '^' for t in type_list]*len(particle_color), dtype=str)
    #particle_marker = np.array(['*' if t == 'ELMT' else ('o' if t != 'M' else '^') for t in type_list] * len(particle_color), dtype=str)
    particle_marker = np.array([
        'D' if t == 'M' else
        '*' if t == 'ELMT' else
        'o' for t in type_list
    ]* len(particle_color), dtype=str)

    #def normalize_size(size):
        #if size <= 30:
            # Use square root scaling for small sizes to emphasize differences
            #return np.sqrt(size) * 4
        #else:
            # Linear scaling for larger sizes, but reduced to avoid overwhelming the plot
            #return size * 0.8

    #sizes = np.array([normalize_size(size) for size in sizes_list])

    # Normalize sizes
    sizes = (np.array(sizes_list, dtype=float) + 1) / (np.max(sizes_list) - np.min(sizes_list)) * 100

    # Build legend elements
    legend_elements = [
        mlines.Line2D([], [],color='None', marker='o', markersize=6, markeredgecolor='red', markerfacecolor='red', linewidth=0, label='C'),
        mlines.Line2D([], [],color='None', marker='o', markersize=6, markeredgecolor='red', linewidth=0, label='C?'),
        mlines.Line2D([], [],color='None', marker='o', markersize=6, markeredgecolor='blue', markerfacecolor='blue', linewidth=0, label='TCN'),
        mlines.Line2D([], [],color='None', marker='o', markersize=6, markeredgecolor='blue', linewidth=0, label='TCN?'),
        mlines.Line2D([], [],color='None', marker='o', markersize=6, markeredgecolor='green', markerfacecolor='green', linewidth=0, label='TCA'),
        mlines.Line2D([], [],color='None', marker='o', markersize=6, markeredgecolor='green', linewidth=0, label='TCA?'),
        mlines.Line2D([], [],color='None', marker='o', markersize=6, markeredgecolor='purple', markerfacecolor='purple', linewidth=0, label='AOS'),
        mlines.Line2D([], [],color='None', marker='o', markersize=6, markeredgecolor='purple', linewidth=0, label='AOS?'),
        mlines.Line2D([], [],color='None', marker='o', markersize=6, markeredgecolor='black', markerfacecolor='black', linewidth=0, label='NA'),
        mlines.Line2D([], [], color='None', marker='D', linestyle='None', markersize=8, linewidth=0, label='Mineral',
                      markerfacecolor='grey', markeredgecolor='black'),
        mlines.Line2D([], [], color='None', marker='*', linestyle='None', markersize=15, linewidth=0, label='Pure element',
                      markerfacecolor='yellow', markeredgecolor='black')
    ]

    # Create legend for sizes
    legend_sizes = np.linspace(np.min(sizes), np.max(sizes), 4)
    legend_labels = np.around(np.linspace(np.min(sizes_list[sizes_list > 0]), np.max(sizes_list), 4), 2)
    legend_elements_bis = list(zip(legend_sizes, legend_labels))

    return particle_color, particle_marker, legend_elements, legend_elements_bis, sizes, particle_edgecolor


def centroids(spectra_list : np.ndarray, name_list : np.ndarray, clusters : np.ndarray, nmb_clusters : int, centroid_number_average : int) -> pd.DataFrame:
    """
    Calculates the centroids of the clusters and find the nearest real spectra to have a look at the average composition.  

    Parameters:
        spectra_list (np.ndarray): An array containing the vectors of the spectra. 
        name_list (np.ndarray): An array containing the names of the particles. 
        clusters (np.ndarray): An array containing the clusters numbers for each particle. 
        nmb_clusters (int): The number of clusters. 
        centroid_number_average (int): The number of spectra near the average to find.

    Returns:
        centroids (pd.DataFrame): A DataFrame containing the centroid spectra, the closest real spectra and its name, for each cluster. 
    """        

    clusters_index = [i for i in range(0, nmb_clusters)]
    centroid_list = []
    closest_spectra_list = []
    closest_spectra_name_list = []


    for nmb in clusters_index:
        # Getting the index of the points
        index = np.where(clusters == nmb)[0]
        cluster_spectra = spectra_list[index]
        cluster_names = name_list[index]
    
        # Calculating the centroid
        centroid = np.mean(cluster_spectra, axis=0, dtype=np.float64)
        centroid_list.append(centroid)

        # Finding the closest real spectra
        distances = np.linalg.norm(cluster_spectra - centroid, axis=1)
        min_distance_index = []

        for i in range(centroid_number_average):
            min_distance = np.argmin(distances)
            min_distance_index.append(min_distance)
            distances = np.delete(distances, min_distance)

        closest_spectra = np.array(cluster_spectra)[min_distance_index]
        closest_spectra_list.append(closest_spectra)
        closest_spectra_name = np.array(cluster_names, dtype=str)[min_distance_index]
        closest_spectra_name_list.append(closest_spectra_name)

    data = {'Cluster index' : clusters_index, 'Centroid' : centroid_list, 'Closest spectra name' : closest_spectra_name_list, 'Closest real spectra' : closest_spectra_list}
    centroids = pd.DataFrame(data=data)

    return centroids

def finding_principal_arrow(xs : np.ndarray, ys : np.ndarray, number_elements : int) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the principal arrows giving the principal composition of PCA.  

    Parameters:
        xs (np.ndarray): The x array obtained by pca.components_. 
        ys (np.ndarray): The y array obtained by pca.components_. 
        number_elements (int): The number of elements wanted.

    Returns:
        tuple:
            x_compo (np.ndarray): The new x array.
            y_compo (np.ndarray): The new y array.
            legend_elements_bis (list): The legend of the graph.
            color_arrow (np.ndarray): The array of color.
    """   
    # Initialization
    x_compo = []
    y_compo = []

    # Each element is associated to a color and a position in the 1536 dimensions vector (according to abscissa energy value)
    element_composition = {
        'Na' : ((0.12156862745098039, 0.4666666666666667, 0.7058823529411765, 1.0), 161),
        'Mg' : ((1.0, 0.4980392156862745, 0.054901960784313725, 1.0), 192),
        'Al' : ((0.17254901960784313, 0.6274509803921569, 0.17254901960784313, 1.0), 227),
        'Si' : ((0.8392156862745098, 0.15294117647058825, 0.1568627450980392, 1.0), 266),
        'S' : ((0.5803921568627451, 0.403921568627451, 0.7411764705882353, 1.0), 355),
        'Cl' : ((0.5490196078431373, 0.33725490196078434, 0.29411764705882354, 1.0), 408),
        'Cd' : ((0.8901960784313725, 0.4666666666666667, 0.7607843137254902, 1.0), 479),
        'K' : ((0.4980392156862745, 0.4980392156862745, 0.4980392156862745, 1.0), 509),
        'Sb' : ((0.7372549019607844, 0.7411764705882353, 0.13333333333333333, 1.0), 546),
        'Sn' : ((0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0), 568),
        'Ca' : ((0.6823529411764706, 0.7803921568627451, 0.9098039215686274, 1.0), 574),
        'Ti' : ((1.0, 0.7333333333333333, 0.47058823529411764, 1.0), 698),
        'Ba' : ((0.596078431372549, 0.8745098039215686, 0.5411764705882353, 1.0), 748),
        'Ce' : ((1.0, 0.596078431372549, 0.5882352941176471, 1.0), 810),
        'Cr' : ((0.7725490196078432, 0.6901960784313725, 0.8352941176470589, 1.0), 838),
        'Fe' : ((0.7686274509803922, 0.611764705882353, 0.5803921568627451, 1.0), 988),
        'Fe (bis)' : ((0.9686274509803922, 0.7137254901960784, 0.8235294117647058, 1.0), 1086),
        'Ni' : ((0.7803921568627451, 0.7803921568627451, 0.7803921568627451, 1.0), 1152),
        'Cu' : ((0.8588235294117647, 0.8588235294117647, 0.5529411764705883, 1.0), 1236)
    }

    xs_update = xs[np.array([value[1] for value in element_composition.values()])]
    ys_update = ys[np.array([value[1] for value in element_composition.values()])]

    norm = np.sqrt(xs_update ** 2 + ys_update ** 2)

    index = np.argsort(norm)[-number_elements:][::-1]

    x_compo = xs_update[index]
    y_compo = ys_update[index]

    # creating a new dictionary
    filtered_element_composition = {list(element_composition.keys())[i]: list(element_composition.values())[i] for i in index}

    # Building the legend of the graph
    legend_elements_bis = [mpatches.Patch(color=color, label=label) for color, label in zip([value[0] for value in filtered_element_composition.values()],[key for key in filtered_element_composition.keys()])]
    
    # Building the color array
    color_arrow = [value[0] for value in filtered_element_composition.values()]
    
    return x_compo, y_compo, legend_elements_bis, color_arrow


def optimize_label_positions_with_bounding_boxes_jax(
    point_positions,
    label_texts,
    overlap_weight=1e2,
    label_dist_weight=1.0,
    init_random_scale=0.0,
    scatter_kwargs=None,
    n_iters=1001,
    learning_rate=5 * 1e-4,
    callback=None,
    plot_initial_positions=False):
    """Optimize label positions in a scatter plot

    Args:
        point_positions (np.ndarray): (n_points, 2) array of point positions
        label_texts (List[str]): list of labels to plot
        overlap_weight (float): weight of the penalty for pairwise intersections
        label_dist_weight (float): weight of the penalty for distance between labels and points
        init_random_scale (float): scale of the initial random perturbation
        scatter_kwargs (dict): keyword arguments passed to the scatter plot
        n_iters (int): number of optimization iterations
        learning_rate (float): learning rate
        callback (Callable): function that is called after each optimization iteration
        plot_initial_positions (bool): whether to plot the initial positions and bounding boxes
    """

    coord_range = point_positions.max(axis=0) - point_positions.min(axis=0)
    normed_point_positions = point_positions / coord_range

    n_pts = normed_point_positions.shape[0]

    fig, ax = plt.subplots()
    if scatter_kwargs is None:
        scatter_kwargs = {}
    scatter = ax.scatter(point_positions[:, 0], point_positions[:, 1], **scatter_kwargs)
    plot_labels(point_positions, point_positions, label_texts)
    plt.draw()

    ax = plt.gca()
    txt_elements = [
        artist
        for artist in ax.get_children()
        if isinstance(artist, matplotlib.text.Text) and len(artist.get_text()) > 0
    ]
    assert len(txt_elements) == n_pts

    text_rel_bboxes = get_text_bounding_boxes(
        ax, txt_elements, point_positions, absolute=False, plot=False
    )
    text_rel_bbox_matrix = jnp.array(text_rel_bboxes)
    xa = 0.01 * coord_range[0]
    ya = 0.01 * coord_range[1]
    bbox_offset = np.array([[-xa, -ya], [xa, ya]])
    text_rel_bbox_matrix += bbox_offset

    scatter_bbox_list = get_scatter_bounding_boxes(ax, scatter)
    scatter_bbox_matrix = jnp.array(scatter_bbox_list)
    scatter_bbox_matrix += bbox_offset

    if plot_initial_positions:
        for i_pt in range(n_pts):
            plot_bounding_box(
                text_rel_bbox_matrix[i_pt, :] + point_positions[i_pt, :], ax=ax
            )
            plot_bounding_box(scatter_bbox_matrix[i_pt, :], ax=ax)

    else:
        plt.close(fig)

    def calculate_objective_components(x_2d):
        # a) distances between points and the respective labels
        label_distance = label_dist_weight * jnp.sum(
            (x_2d - normed_point_positions) ** 2
        )

        # b) pairwise intersections between label bounding boxes
        text_bbox_matrix = text_rel_bbox_matrix + jnp.swapaxes(
            (x_2d * coord_range).reshape((x_2d.shape[0], 2, 1)), 1, 2
        )
        interlabel_inters_matrix = multiple_bbox_intersections(
            text_bbox_matrix, text_bbox_matrix, coord_range=coord_range
        )
        interlabel_inters_array = interlabel_inters_matrix[np.triu_indices(n_pts, 1)]
        interlabel_inters_loss = overlap_weight * jnp.sum(interlabel_inters_array)

        # c) distances between labels and point bounding boxes
        label_point_inters_matrix = multiple_bbox_intersections(
            text_bbox_matrix, scatter_bbox_matrix, coord_range=coord_range
        )
        label_point_inters_array = label_point_inters_matrix.reshape(-1)
        label_point_inters_loss = overlap_weight * jnp.sum(label_point_inters_array)

        obj_components = {
            "label_distance": label_distance,
            "interlabel_inters_loss": interlabel_inters_loss,
            "label_point_inters_loss": label_point_inters_loss,
        }
        return obj_components

    def fun_to_minimize(x_2d):
        obj_components = calculate_objective_components(x_2d)
        return (
            obj_components["label_distance"]
            + obj_components["interlabel_inters_loss"]
            + obj_components["label_point_inters_loss"]
        )

    x_init = normed_point_positions
    if init_random_scale > 0:
        x_init = x_init + np.random.normal(0, init_random_scale, (n_pts, 2))
    x_2d = x_init

    optimizer = optax.adam(learning_rate=learning_rate)
    opt_state = optimizer.init(x_2d)

    @jax.jit
    def perform_optim_step(x_2d, opt_state):
        """Do one gradient-based optimization step"""
        loss_value, grads = jax.value_and_grad(fun_to_minimize)(x_2d)
        updates, opt_state = optimizer.update(grads, opt_state)
        x_2d = optax.apply_updates(x_2d, updates)
        return x_2d, opt_state, loss_value, grads

    for i_iter in range(n_iters):
        x_2d, opt_state, loss_value, grads = perform_optim_step(x_2d, opt_state)
        if callback is not None:
            callback(i_iter, loss_value, x_2d, grads)

    opt_label_positions = x_2d * coord_range

    return opt_label_positions


def multiple_bbox_intersections(bbox_matrix, other_bbox_matrix, coord_range=(1, 1)):
    """Calculate the pairwise intersections of two sets of bounding boxes

    This vectorized implementation is more efficient than the avoided double for loop

    Args:
        bbox_matrix: numpy array of shape (n, 2, 2)
            dimensions: points, min and max, xy coordinates
        other_bbox_matrix: numpy array of shape (m, 2, 2)
            dimensions: points, min and max, xy coordinates
        coord_range: tuple of floats representing the coordinate range (x, y)

    Returns:
        numpy array of shape (n, m)
    """
    rep_bbox_matrix = jnp.repeat(bbox_matrix, other_bbox_matrix.shape[0], axis=0)
    rep_other_bbox_matrix = jnp.tile(other_bbox_matrix, (bbox_matrix.shape[0], 1, 1))

    coord_max = rep_bbox_matrix[:, 1, :]
    coord_min = rep_bbox_matrix[:, 0, :]
    coord_max_other = rep_other_bbox_matrix[:, 1, :]
    coord_min_other = rep_other_bbox_matrix[:, 0, :]
    intersects = jnp.clip(
        coord_max - coord_min_other, a_max=coord_max_other - coord_min
    )

    intersect_prods = (
        jnp.clip(intersects[:, 0], 0, np.inf)
        / coord_range[0]
        * jnp.clip(intersects[:, 1], 0, np.inf)
        / coord_range[1]
    )
    return intersect_prods.reshape(bbox_matrix.shape[0], -1)