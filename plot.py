import h5py
import matplotlib.pyplot as plt

f = h5py.File('/kaggle/working/result/test_Unet/reconstructions_val/input/brain_acc4_179.h5', 'r')

input_acc4 = f['input']
recon_acc4 = f['reconstruction']
target_acc4 = f['target']

plt.figure(figsize=(30, 30))
plt.subplot(1, 3, 1)
plt.imshow(input_acc4[1, :, :])
plt.title('input_acc4')
plt.subplot(1, 3, 2)
plt.imshow(recon_acc4[1, :, :])
plt.title('reconstruction_acc4')
plt.subplot(1, 3, 3)
plt.imshow(target_acc4[1, :, :])
plt.title('target_acc4')
#plt.savefig('input_result_acc4.png', dpi=300)


f = h5py.File('/kaggle/working/result/test_Unet/reconstructions_val/grappa/brain_acc4_179.h5', 'r')

input_acc4 = f['input']
recon_acc4 = f['reconstruction']
target_acc4 = f['target']

plt.figure(figsize=(30, 30))
plt.subplot(2, 3, 1)
plt.imshow(input_acc4[1, :, :])
plt.title('grappa_acc4')
plt.subplot(2, 3, 2)
plt.imshow(recon_acc4[1, :, :])
plt.title('reconstruction_acc4')
plt.subplot(2, 3, 3)
plt.imshow(target_acc4[1, :, :])
plt.title('target_acc4')
#plt.savefig('grappa_result_acc4.png', dpi=300)


f = h5py.File('/kaggle/working/result/test_Unet/reconstructions_val/input/brain_acc8_187.h5', 'r')
input_acc8 = f['input']
recon_acc8 = f['reconstruction']
target_acc8 = f['target']

plt.figure(figsize=(30, 30))
plt.subplot(3, 3, 1)
plt.imshow(input_acc8[1, :, :])
plt.title('input_acc8')
plt.subplot(3, 3, 2)
plt.imshow(recon_acc8[1, :, :])
plt.title('reconstruction_acc8')
plt.subplot(3, 3, 3)
plt.imshow(target_acc8[1, :, :])
plt.title('target_acc8')
#plt.savefig('input_result_acc8.png', dpi=300)


f = h5py.File('/kaggle/working/result/test_Unet/reconstructions_val/grappa/brain_acc8_187.h5', 'r')
input_acc8 = f['input']
recon_acc8 = f['reconstruction']
target_acc8 = f['target']

plt.figure(figsize=(30, 30))
plt.subplot(4, 3, 1)
plt.imshow(input_acc8[1, :, :])
plt.title('grappa_acc8')
plt.subplot(4, 3, 2)
plt.imshow(recon_acc8[1, :, :])
plt.title('reconstruction_acc8')
plt.subplot(4, 3, 3)
plt.imshow(target_acc8[1, :, :])
plt.title('target_acc8')
#plt.savefig('grappa_result_acc8.png', dpi=300)
##
plt.show()
