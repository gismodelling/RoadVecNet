EPOCHS = 100
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,EPOCHS), history.history["loss"], linewidth=2, color='blue', label="train_loss")
plt.plot(np.arange(0,EPOCHS), history.history["val_loss"], linewidth=2,  color='red', label="val_loss")
#plt.plot(np.arange(0,EPOCHS), history.history["accuracy"], linewidth=2, color='blue', label="train_acc")
#plt.plot(np.arange(0,EPOCHS), history.history["val_accuracy"], linewidth=2,  color='red', label="val_acc")
plt.title("Training and Validation Losses", fontsize=14)
plt.xlabel("Epoch", fontsize=13, color='black')
plt.ylabel("Loss", fontsize=13, color='black')
plt.tick_params(labelsize=11, size = 5, rotation=0, color = 'black', labelcolor = 'black')
leg = plt.legend(loc="best", fontsize=12)
leg.get_frame().set_edgecolor('black')
