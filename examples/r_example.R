# Insight137 EAP — R Integration Example
# Load results exported from Python library (CSV format)

# Read the CSV exported from Python:
# eap.to_csv(profiles, 'eap_results.csv', labels=['human', 'bot'])
data <- read.csv('eap_results.csv')

# View profiles
print(data)

# Radar chart using fmsb package
# install.packages("fmsb")
library(fmsb)

# Prepare for radar chart (fmsb needs max/min rows)
radar_data <- rbind(
  rep(max(data[,2:5]), 4),  # max
  rep(0, 4),                 # min
  data[,2:5]                 # actual values
)
colnames(radar_data) <- c("Psi1", "Psi2", "Psi3", "Psi4")

colors <- c(rgb(0.31, 0.80, 0.77, 0.5), rgb(0.96, 0.62, 0.04, 0.5))
radarchart(radar_data, pcol=colors, pfcol=colors, plwd=2,
           title="EAP Psi Profile Comparison")
legend("topright", legend=data$label, col=colors, lwd=2)
