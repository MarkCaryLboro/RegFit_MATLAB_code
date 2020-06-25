# MATLAB_RegFit_Code
MATLAB RIGLS Algorithm

by

Mark Cary

Implements general nonlinear regularised iterative least squares algorithm, with optimal hyper-parameter selection.

Hyper-parameter selection based on fixed point iteration at each step of the nonlinear search.
Hyper-parameter selection based on information theoretic criterion, either AICc or BIC.
Supports heteroscedastic or iid data.
Can add in any custom fit model or covariance model.
Can implement custom information criterion for hyper-parameter selection.
Supports bootstrapping and confidence interval generation.
Model diagnostic plots created.
User documentation available, including installation instructions.
Issues

hasm class not complete
Does not support serially correlated data
Future Enhancements

Support serially corellated data for time series analysis
Implement block bootstrapping tool
Add higher order regularisation functionals

Version 2.0 Enhancement 04/06/2020 by M. Cary

Added bspm (B-spline model class). Data must be coded onto the interval X--> [0, 1]. It is recommended Y-data are coded onto 
the same interval. RegFitUserNotes.doc has been updated to reflect bspm usage.

Version 3.0 Changes and Enhancement 20/06/2020 by M. Cary

Removed context layer in the code to simplify architecture. Plotting now at
the correct level - moved to regNonlinIGLS class. Interface is more streamlined
and compact.

Added truncated power series spline model.

Added truncated power series spline model for SoH predictive purposes. 