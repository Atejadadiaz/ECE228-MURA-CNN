import os

if not os.path.exists("MURA-v1.1") and os.path.exists("MURA-v1.1.zip"):
    print("Unzipping MURA dataset...")
    os.system("unzip -u MURA-v1.1.zip")
    print("Done.")
else:
