# Predicting Streamflow Using Machine Learning in Northern Canada

Abstract:
----
Predicting streamflow under varying climate conditions is critically important for managing water resources. This is especially true as climate change intensifies the hydrologic cycle in certain regions, such as the northern latitudes. In the past, physical models have been used to predict streamflow based on complex equations describing how water moves across a landscape. These models are difficult to implement and require large amounts of data for validation that is not always available, especially in remote, northern watersheds. Machine learning offers a new way to predict streamflow without significant expertise with physical modeling. In this study, we demonstrate the utility of three different machine learning methods - random forest, multilayer perceptron, and long short term memory – to predict streamflow for three different river gauges in northern Canada. We find that the random forest method produces the best predictions, with high accuracies >90% for all gauges. We emphasize the need for larger training datasets to build more robust models.
----

To run our models in a collaborative, shared environment, we used a virtual machine (VM) on the Google Cloud Platform (GCP). Our VM had 16 CPUs, 60 GB of RAM, and one NVIDIA Tesla T4 GPU. This VM was not really necessary for this project (we did not utilize the GPU), but it was a useful exercise to create a VM and run code via the command line and Ubuntu Desktop. The cost was $0.76 per hour to run this VM, and we used Google Research Credits to pay for it.

This project is organized as follows:
- Sarah's Code: Sarah's Random Forest training scripts
- Katia's Code: Katia's Multilayer Perceptron and LSTM training notebook
- Preprocessing_Data.ipynb: data preprocessing/cleaning notebook
- Figures.ipynb: figure making notebook

Thanks!

