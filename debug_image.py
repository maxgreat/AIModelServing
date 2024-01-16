from PIL import Image

im = Image.open('incorrect.gif')
im.seek(0)
corrected = im.copy()
corrected.putalpha(255)
corrected = corrected.convert('RGBA')
list_images = []
for i in range(1, im.n_frames):
    im.seek(i)
    c = im.copy()
    #c.putalpha(255)
    list_images.append(c.convert('RGBA'))
print("List images :", list_images)
corrected.save('corrected.gif', save_all=True, append_images=list_images, optimize=False, loop=0)