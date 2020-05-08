{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "from bokeh.io import curdoc, output_notebook, output_file, show, save\n",
    "from bokeh.plotting import figure, show\n",
    "from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper, Slider, PreText, CheckboxButtonGroup, WheelZoomTool\n",
    "from bokeh.palettes import Spectral6, brewer\n",
    "from bokeh.layouts import widgetbox, row, column, layout\n",
    "from bokeh.models import (CDSView, ColorBar, ColumnDataSource,\n",
    "                          CustomJS, CustomJSFilter, \n",
    "                          GeoJSONDataSource, HoverTool,\n",
    "                          LinearColorMapper, Slider, Paragraph,\n",
    "                          Legend, LegendItem, Label,FactorRange)\n",
    "from bokeh.models.widgets import Panel, Tabs\n",
    "from bokeh.tile_providers import CARTODBPOSITRON, get_provider\n",
    "from bokeh.transform import jitter\n",
    "from bokeh.embed import file_html\n",
    "from bokeh.resources import CDN\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "from datetime import datetime\n",
    "import datetime as dt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "df_crash = pd.read_csv(\"C:\\\\Users\\Rimmen\\crash_data.csv\", low_memory = False)\n",
    "df_crash=df_crash.sample(100000)\n",
    "df_crash.drop(df_crash.loc[:, 'CONTRIBUTING FACTOR VEHICLE 1':'VEHICLE TYPE CODE 5'].columns, axis = 1, inplace = True) \n",
    "df_crash.drop(df_crash.loc[:, 'ON STREET NAME':'OFF STREET NAME'].columns, axis = 1, inplace=True)\n",
    "df_crash"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "\n",
    "df_crash = df_crash.loc[(df_crash['NUMBER OF PERSONS INJURED'] > 0) | (df_crash['NUMBER OF PERSONS KILLED'] > 0 )]  \n",
    "df_crash = df_crash.loc[(df_crash['LATITUDE'].notna() & df_crash['LONGITUDE'].notna() & df_crash['LONGITUDE']!=0 & (df_crash['LATITUDE'] != 0 )) & df_crash['BOROUGH'].notna() & (df_crash['LATITUDE']<41)].copy()\n",
    "df_crash[\"CRASH DATE\"] = df_crash[\"CRASH DATE\"].apply(lambda x: datetime.strptime(x, \"%Y-%m-%d\") )\n",
    "df_crash[\"CRASH TIME\"] = df_crash[\"CRASH TIME\"].apply(lambda x: datetime.strptime(x, \"%H:%M:%S\") )\n",
    "df_crash['Year'] = df_crash['CRASH DATE'].dt.year\n",
    "df_crash['Month'] = df_crash['CRASH DATE'].dt.month\n",
    "df_crash['Hour'] = df_crash['CRASH TIME'].dt.hour\n",
    "df_crash['Weekday'] = df_crash['CRASH DATE'].dt.day_name()\n",
    "\n",
    "\n",
    "#Preparing data for tab1\n",
    "#Converting coordinates\n",
    "def wgs84_to_web_mercator(df_crash, lon=\"LONGITUDE\", lat=\"LATITUDE\"):\n",
    "    k = 6378137\n",
    "    df_crash[\"x\"] = df_crash[lon] * (k * np.pi/180.0)\n",
    "    df_crash[\"y\"] = np.log(np.tan((90 + df_crash[lat]) * np.pi/360.0)) * k\n",
    "    return df_crash\n",
    "wgs84_to_web_mercator(df_crash)\n",
    "\n",
    "#Picking out data with Borough and lat/long\n",
    "\n",
    "#Filtering crashes with injuries/deaths \n",
    "\n",
    "xmin, xmax = min(df_crash['x']), max(df_crash['x'])\n",
    "ymin, ymax = min(df_crash['y']), max(df_crash['y'])\n",
    "\n",
    "\n",
    "borough_data = df_crash.groupby([df_crash['Year'],df_crash['BOROUGH']]).agg({'NUMBER OF PERSONS KILLED': 'sum','NUMBER OF PERSONS INJURED':'sum', \n",
    "                                                                            'NUMBER OF PEDESTRIANS KILLED':'sum','NUMBER OF CYCLIST KILLED':'sum',\n",
    "                                                                            'NUMBER OF MOTORIST KILLED':'sum','NUMBER OF PEDESTRIANS INJURED':'sum',\n",
    "                                                                            'NUMBER OF CYCLIST INJURED':'sum','NUMBER OF MOTORIST INJURED':'sum'}).reset_index()\n",
    "#borough_data = borough_data.loc[(borough_data['Year']==2013) & (borough_data['NUMBER OF PERSONS INJURED'] == 0) & (borough_data['NUMBER OF PERSONS KILLED'] == 0 )]\n",
    "borough_data['PEDESTRIANS'] = borough_data['NUMBER OF PEDESTRIANS KILLED'] + borough_data['NUMBER OF PEDESTRIANS INJURED']\n",
    "borough_data['CYCLISTS'] = borough_data['NUMBER OF CYCLIST KILLED'] + borough_data['NUMBER OF CYCLIST INJURED']\n",
    "borough_data['MOTORISTS'] = borough_data['NUMBER OF MOTORIST KILLED'] + borough_data['NUMBER OF MOTORIST INJURED']\n",
    "borough_data['KILLED'] = borough_data['NUMBER OF PEDESTRIANS KILLED'] + borough_data['NUMBER OF CYCLIST KILLED'] + borough_data['NUMBER OF MOTORIST KILLED']  \n",
    "borough_data['INJURED'] = borough_data['NUMBER OF PEDESTRIANS INJURED'] + borough_data['NUMBER OF CYCLIST INJURED'] + borough_data['NUMBER OF MOTORIST INJURED']\n",
    "\n",
    "#data =hurra.loc[(hurra['Year']==2013) & (hurra['NUMBER OF PERSONS INJURED'] == 0) & (hurra['NUMBER OF PERSONS KILLED'] == 0 )\n",
    "source2 = ColumnDataSource(data =borough_data)\n",
    "source = ColumnDataSource(data = df_crash)\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "#PLOT OF MAP \n",
    "\n",
    "#source = ColumnDataSource(data = df)\n",
    "plot = figure(plot_height = 750, plot_width = 750,x_axis_type=\"mercator\", y_axis_type=\"mercator\",x_range = (xmin-1000,xmax+1000),y_range=(ymin-1000,ymax+1000))\n",
    "plot.axis.visible = False\n",
    "tile_provider = get_provider(CARTODBPOSITRON)\n",
    "plot.add_tile(tile_provider)\n",
    "plot.circle(x='x', y='y', fill_alpha = 0.05, source = source, legend_label = 'Injury/Death', color = 'red', size=3)\n",
    "plot.toolbar.active_scroll = plot.select_one(WheelZoomTool)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# PLOT OF BOROUGHS (VBAR)\n",
    "\n",
    "#source2 = ColumnDataSource(data =df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "boroughs = df_crash['BOROUGH'].unique()\n",
    "road_user = ['PEDESTRIANS', 'CYCLISTS','MOTORISTS']\n",
    "p = figure(x_range = boroughs,plot_width=400, y_range = (0,100), plot_height=500, tools = \"hover\", tooltips = \"$name: @$name\")\n",
    "p.vbar_stack(road_user, x='BOROUGH', width=0.5, color=(\"brown\", \"blue\",\"lightblue\"), source=source2, legend_label =road_user);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "#LEGENDS (As text)\n",
    "\n",
    "label1 = \"0\"\n",
    "label2 = \"0\"\n",
    "legend_test = figure(plot_width=400, plot_height=100,tools=\"\")\n",
    "legend_test.circle([0], [0], size=20, color=\"olive\", alpha=0.0)\n",
    "legend_test.axis.visible = False\n",
    "legend = Legend(items=[\n",
    "    LegendItem(label=\"Deaths:0\", index=0),\n",
    "    LegendItem(label=\"Injuries:0\", index=1),\n",
    "])\n",
    "\n",
    "legend_test.toolbar.logo = None\n",
    "legend_test.xgrid.visible = False\n",
    "legend_test.ygrid.visible = False\n",
    "legend_test.add_layout(legend)\n",
    "legend_test.legend.label_text_font_size = \"20px\"\n",
    "legend_test.legend.location = \"center_left\"\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Jitterplot - Time\n",
    "DAYS = ['Sunday', 'Saturday', 'Friday', 'Thursday', 'Wednesday', 'Tuesday', 'Monday']\n",
    "p2 = figure(plot_width=1150, plot_height=300, y_range=DAYS, x_axis_type='datetime', \n",
    "           title=\"Commits by Time of Day (US/Central) 2012—2016\")\n",
    "\n",
    "p2.circle(x='CRASH TIME', y=jitter('Weekday', width=0.4, range=p2.y_range),  source=source, alpha=0.2)\n",
    "\n",
    "p2.xaxis[0].formatter.days = ['%H:%M']\n",
    "p2.x_range.range_padding = 0\n",
    "p2.ygrid.grid_line_color = None"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Slider & Checkboxs\n",
    "checkbox_labels = ['Injuries','Deaths']\n",
    "checkbox_button_group = CheckboxButtonGroup(\n",
    "        labels=checkbox_labels, active=[])\n",
    "slider = Slider(start=2013, end = 2019, step = 1, value = 2013, title = 'Year')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [],
   "source": [
    "#function handler\n",
    "def update_plot(attr, old, new):\n",
    "    checkbox_values = [checkbox_button_group.labels[i] for i in checkbox_button_group.active]\n",
    "    yr = slider.value\n",
    "    new_data = df_crash\n",
    "    new_data2 = borough_data\n",
    "    #new_data2 = df.groupby([df['Year'],df['BOROUGH']]).agg({'NUMBER OF PERSONS KILLED': 'sum','NUMBER OF PERSONS INJURED':'sum', 'NUMBER OF PEDESTRIANS KILLED':'sum','NUMBER OF CYCLIST KILLED':'sum','NUMBER OF MOTORIST KILLED':'sum','NUMBER OF PEDESTRIANS INJURED':'sum','NUMBER OF CYCLIST INJURED':'sum','NUMBER OF MOTORIST INJURED':'sum'}).reset_index()\n",
    "    #Filtering year\n",
    "    #legend_test.add_layout(legend)\n",
    "    #MAPPING DATA\n",
    "    if ('Injuries' in checkbox_values) & ('Deaths' in checkbox_values):        \n",
    "        new_data = new_data.loc[(new_data['NUMBER OF PERSONS INJURED'] > 0) | (new_data['NUMBER OF PERSONS KILLED'] > 0 )]\n",
    "    elif 'Injuries' in checkbox_values:\n",
    "        new_data = new_data.loc[new_data['NUMBER OF PERSONS INJURED'] > 0]\n",
    "    elif 'Deaths' in checkbox_values:\n",
    "        new_data = new_data.loc[new_data['NUMBER OF PERSONS KILLED'] > 0]\n",
    "    else:\n",
    "        new_data = new_data.loc[(new_data['NUMBER OF PERSONS INJURED'] == 0) & (new_data['NUMBER OF PERSONS KILLED'] == 0 )]\n",
    "\n",
    "    #VBAR DATA\n",
    "    if ('Injuries' in checkbox_values) & ('Deaths' in checkbox_values):        \n",
    "        p.y_range.end = new_data2[['KILLED','INJURED']].max().max()\n",
    "    elif 'Injuries' in checkbox_values:\n",
    "        p.y_range.end = new_data2['INJURED'].max()        \n",
    "    elif 'Deaths' in checkbox_values:\n",
    "        p.y_range.end = new_data2['KILLED'].max()\n",
    "    else:\n",
    "        p.y_range.end = 100\n",
    "  \n",
    "    new_data = new_data.loc[new_data['Year']==yr].copy()\n",
    "    new_data2 = new_data2.loc[new_data2['Year']==yr].copy()\n",
    "  \n",
    "    #Textbox\n",
    "    label1 = f\"Deaths:{new_data2['KILLED'].sum()}\"\n",
    "    label2 = f\"Injuries:{new_data2['INJURED'].sum()}\"\n",
    "    legend_test.legend.items = [LegendItem(label =label1, index = 0),LegendItem(label = label2,index=1),]\n",
    "  \n",
    "    #DATA\n",
    "    source.data = new_data\n",
    "    source2.data = new_data2\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "#On Change\n",
    "checkbox_button_group.on_change('active',update_plot)    \n",
    "slider.on_change('value', update_plot)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Layout\n",
    "plots2 = column(checkbox_button_group,slider,legend_test,p)\n",
    "l1 = layout([\n",
    "    [plots2,plot],\n",
    "    [p2]\n",
    "])\n",
    "tab1 = Panel(child=l1, title=\"hola\")\n",
    "\n",
    "tabs = Tabs(tabs=[tab1])\n",
    "curdoc().add_root(tabs)\n",
    "#curdoc().title =\"Sliders\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "metadata": {},
   "outputs": [],
   "source": [
    "#output_file(\"js_events.html\", title=\"JS Events Example\")\n",
    "#show(tabs)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
