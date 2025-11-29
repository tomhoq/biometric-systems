 Here is a small description of what each folder/file was used for:

|
|
|---- hugging_face_dataset.ipynb
|       - used to perform image analysis on data, but also to experiment. Everything I did there was used as a basis for the train and evaluate scripts in src/
|
|
|---- out/
|       - stores all teh outputs from each run. This includes evaluation metrics, model saved, scores computed on evaluation for each dataset variation
|       - Disclaimer: The Run names assigned in out/ differ from the ones shown in the paper. To see the mapping open the sheet-scores-new.xlsx
|
|
|---- src/
|       - scripts used to train and evaluate the models
|
|
|
|---- job_scripts/all.sh
|       - script to submit a job in the hpc. It first trains then evaluates using the scripts in src/
|
|
|
|---- det_curve_report
|       - contains the scores used to recreate the DET curve shown in the paper
|
|---- datasets/ and processed_datasets/ 
|       - stores the dataset in hugging face dataset format. Processed dataset was converted into pixel values and is ready to be used for training/inference 
|
|____ sheet-scores-new.xlsx 
        - stores all teh scores and configurations used in an excel sheet. There is more than one page.


I believe the rest is pretty self-explanatory.

NOTE:
All the datasets / processed datasets were emptied as they simply occupied too much space in the zip. This means some part of the notebooks wont run.
Also all the models inside each run had to be deleted due to the zip becoming too big

