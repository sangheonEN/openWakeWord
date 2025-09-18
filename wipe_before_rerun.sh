# Remove all generated/training artifacts
# rm -rf ./generated_clips/
rm -rf ./my_custom_model/
rm -f ./my_model.yaml
# rm -f ./test_generation.wav

# Remove any .npy files that aren't the downloaded features
# (be careful not to delete the downloaded validation/training features)
# find . -name "*.npy" -not -name "openwakeword_features_ACAV100M_2000_hrs_16bit.npy" -not -name "validation_set_features.npy" -delete