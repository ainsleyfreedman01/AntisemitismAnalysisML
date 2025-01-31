import pandas as pd
import matplotlib.pyplot as plt

# load the dataset
df = pd.read_csv('data/campus_reports.csv')

incident_type_counts = df['Incident Type'].value_counts()


incident_type_percentages = (incident_type_counts / incident_type_counts.sum()) * 100


plt.figure(figsize=(7, 7))
colors = plt.cm.Paired.colors
plt.pie(incident_type_counts, startangle=90, colors=colors)


legend_labels = [f"{incident_type} ({percent:.1f}%)" for incident_type, percent in zip(incident_type_counts.index, incident_type_percentages)]
plt.legend(legend_labels, title="Incident Type", loc="upper left", fontsize=10)


plt.title('Distribution of Incidents by Type', fontsize=12)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.savefig('visualizations/pie_chart.png', bbox_inches='tight', dpi=300)
plt.show()


