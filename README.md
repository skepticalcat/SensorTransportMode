#### Transportmode detection using smartphone sensors 

Uses the SHL Dataset for User1 - Hips.\
Loosely based on the work of Qin et al. [1] \
Achieves approx. 85% - 90% point based accuracy for all modes.\
Utilizes a CNN and a LSTM for classification.

\
Written out of personal interest

##### How to run
Have ~16 GB ram and a CUDA Card.

Run FeatureBuilder to prepare raw data to tensors:

        python FeatureBuilder.py path/to/SHLDataset_User1Hips_v1/release/User1 sample_rate cnn_window_size_sec lstm_window_size_sec num_of_files_to_load boolean_force_reload
        
        Example:
        python FeatureBuilder.py /home/xyz/SHLData/User1 25 3 600 50 False

force_reload determines if the raw_data pickle should be written anew, num_of_files_to_load determines how many sensor folders (days) are to be loaded from the dataset.
This creates two pickles, one with the cleaned raw data, one with the samples to be fed to the NN.

Afterwards, run TransModeDetect.py:

        python TransModeDetect.py lstm_examples_25_3_10 3 600

where the first arg is the name of the tensor pickle, then the cnn window size in seconds and the lstm window size in seconds.

##### References
[1] Qin, Yanjun, et al. "Toward transportation mode recognition using deep convolutional and long short-term memory recurrent neural networks." IEEE Access 7 (2019): 142353-142367.

[2] H. Gjoreski, M. Ciliberto, F. J. Ordoñez Morales, D. Roggen, S. Mekki, S. Valentin. “A versatile annotated dataset for multimodal locomotion analytics with mobile devices.” In ACM Conference on Embedded Networked Sensor Systems. ACM, 2017.