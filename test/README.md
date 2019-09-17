# Cognitive memory with resnet

ResNet implementation was adopted from https://github.com/akamaster/pytorch_resnet_cifar10 (author is Yerlan Idelbayev)

Due to the size of intermediate layers, the codes were tweaked to work step-by-step to minimize the necessary memory.

1. load_model.py: load pretrained model and feed training set to get images of 5 intermediate layers (the first CNN, layer1,2, 3 and FC). layer1, 2 and 3 are composite layers, not single layers. That is why ResNet actually havs tens of layers. 
2. load_image.py: how many novel images exist in all five intermediate layers. 
3. write_pred.py: calculate and save predictions of ResNet on test and training examples. 
4. gen_test_image.py: the same as load_model, but it work with test set, not training set
5. gen_association.py: generate correlation-like association between cognitive memeory of each layer and prediction of ResNet
6. three-pred.py: test how well cognitive memory of each layer can predict the answers of ResNet.
7. load_image_layer.py: calculates cognitive memory for a selective layer over multiple threshold values. 
8. gen_adv.py: generates adversarial examples
9. adv_correlation.py: estimates the consistency between layers
10. plot_results.py: plots consistency values and estimates AUROC.




