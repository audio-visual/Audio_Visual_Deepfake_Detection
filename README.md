# Audio_Visual_Deepfake_Detection

This repository contains the solution for the 2024 1M-Deepfakes Detection Challenge, where we achieved **3rd** place in the temporal localization track.

For more details, please visit: https://github.com/ControlNet/AV-Deepfake1M 

**Pretrained Models Utilized:**

- **Visual Modality:** Lavdf
- **Audio Modality:** BYOL-A, Emotion2Vec

**Conclusions:**

We primarily employed the UMMAFormer method:

1. **Dataset Module Enhancements:**
   - The main modifications were made to the dataset module. Incorperating more rich features.

2. **Accelerated Feature Extraction:**
   - To expedite feature extraction, we implemented batch processing for the entire procedure.
   - Recognizing that different models produce features of varying lengths (i.e., different frame counts), we applied feature-level interpolation to achieve a fixed length in time dimension. Experimental results indicated that this approach outperformed padding with fixed values.

3. **Post-Processing Modification:**
   - We observed that filtering the final localization results to empty list where the confidence scores is below 0.2 can significantly improved performance.



**Final result**

|Method | Modality | Score|
|-------|----------|------|
|UMMAFormer | AV  | 32.499|
|Ours| A| 80.000|
|Ours| AV| 85.218|

**Note:**

Due to the substantial size of the competition dataset, time constraints, and limited resources, we were unable to thoroughly organize the code. As a result, the current code may not be directly executable and is provided for reference purposes only.


