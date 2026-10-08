
# ============================================================
# Partial Least Squares Structural Equation Modeling (PLS-SEM)
# ============================================================
# This script examines developmental associations among age,
# event-related CAP state engagement, Theory of Mind (ToM),
# and empathic ability in children.
#
# Analysis workflow:
# 1. Specify the measurement model using observed indicators
#    of age, CAP state engagement, ToM, and empathy.
# 2. Define the hypothesized structural paths among constructs.
# 3. Estimate the PLS-SEM with 5,000 bootstrap resamples.
# 4. Extract path coefficients and bootstrap statistics.
#
# Input: Preprocessed subject-level data containing all
# required indicators.
# ============================================================

library(plspm)


# ============================================================
# Step 1: Load prepared data
# ============================================================
# The input data contain subject-level behavioral measures
# and event-related CAP occurrence differences (social - control).
# All indicator construction and missing-data handling should
# be completed before running this analysis.

data <- read.csv("data/PLS_input.csv")


# ============================================================
# Step 2: Specify the measurement model
# ============================================================
# Define manifest variables for the six latent constructs.
# CAP state engagement is represented by occurrence differences
# during ToM and Pain events relative to Control events.

blocks <- list(
  c("Age", "Age_squ"),
  c("TOM_Total"),
  c("Empathy_Cognitive", "Empathy_Affective"),
  c("Mental1", "Pain1"),
  c("Mental2", "Pain2"),
  c("Mental3", "Pain3"),
)


# ============================================================
# Step 3: Specify the structural model
# ============================================================
# Rows represent dependent constructs, and columns represent
# their predictors. A value of 1 specifies a directed path.

constructs <- c("Age","TOM", "Empathy",
                "State1", "State2", "State3")

path_matrix <- matrix(0, nrow = 6, ncol = 6)
rownames(path_matrix) <- colnames(path_matrix) <- constructs


path_matrix["State1", "Age"] <- 1
path_matrix["State2", "Age"] <- 1
path_matrix["TOM", "Age"] <- 1
path_matrix["State3", "State1"] <- 1
path_matrix["State2", "State3"] <- 1
path_matrix["State2", "TOM"] <- 1
path_matrix["Empathy", "State3"] <- 1


# ============================================================
# Step 4: PLS-SEM estimation and bootstrap inference
# ============================================================
# Estimate the structural model using Mode A measurement
# blocks and standardized indicators.
# Bootstrap resampling (5,000 iterations) is used to assess
# the uncertainty of the estimated path coefficients.

set.seed(1234)

pls_result <- plspm(
  data,
  path_matrix,
  blocks,
  modes = rep("A", 6),
  scaled = TRUE,
  boot.val = TRUE,
  br = 5000
)


# ============================================================
# Step 5: Extract bootstrap path statistics
# ============================================================
# Extract original path coefficients, bootstrap estimates,
# standard errors, and percentile confidence intervals.

path_summary <- pls_result$boot$paths

# Calculate approximate two-sided p-values using a normal
# approximation based on bootstrap means and standard errors.
path_summary$z_value <- path_summary$Mean.Boot / path_summary$Std.Error
path_summary$p_value <- 2 * pnorm(-abs(path_summary$z_value))

# Assign significance labels based on approximate p-values.
path_summary$Significance <- ifelse(
  path_summary$p_value < 0.001, "***",
  ifelse(path_summary$p_value < 0.01, "**",
         ifelse(path_summary$p_value < 0.05, "*", ""))
)

# Retain the statistics reported for each structural path.
path_summary <- path_summary[, c(
  "Original", "Mean.Boot", "Std.Error",
  "perc.025", "perc.975", "p_value", "Significance"
)]

print(path_summary)
