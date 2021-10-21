# Medical-Image-Segmentation: 2018 Data Science Bowl 
A case study on Nucleus Segmentation across imaging experiments using Deep CNN based models (UNet, UNet++, HRNet).

* Please find a detailed blog on this case study here: https://medium.com/@kriz17/medical-image-segmentation-3093bef449a5 
* Web Application: https://share.streamlit.io/kriz17/mdt_app
### Background

To find the cure for any disease, researchers analyze how the cells of a sample react to various treatments and understand the underlying biological processes. Identifying the cells' nuclei is the starting point for most analyses since it helps identify each individual cell in a sample. Automating this process can save a lot of time, allowing for more efficient drug testing and unlocking the cures faster. 

### Project Summary
In this case study, I propose the use of Deep CNNs for the automation of nucleus detection in images across varied conditions. The data used was from the Kaggle Competetion [2018 Data Science Bowl](https://www.kaggle.com/c/data-science-bowl-2018). I design a total of three networks derived from U-Net, U-Net++, and HRNet and compare their performances using Mean IoU as the metric. It's seen that U-Net based model performs the best with a Mean IoU score of 0.861. Further, I used Float16 quantization to reduce the model size from 28 MB to 14 MB. The average inference time of the model was 0.126 seconds per sample which is decent for the task at hand. Finally, I deployed the model using streamlit. 

### Files Details
* EDA1.ipynb and EDA2.ipynb notebooks contains basic data analysis. 
* Modeling_Unet.ipynb contains the code for the whole pipeline. It contains the code for the UNet based model used along with the experimentation, as well as the code for HRNet 
* Modeling_Unet++.ipynb contains the code for UNet++ based model along with the full comparision of all the models.
* Quantization.ipynb contains the code for the model quantization and its analysis. 

