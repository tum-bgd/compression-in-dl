# Datasets:
# * P1 - AID         JPG75
# * (Selected) P2 - EuroSAT     JPG75
# * (Selected) P3 - Gaofen      PNG75
# * P4 - Resisc45    JPG75
# * (Selected) P5 - RSI-CB256   TIF-None
# * P6 - UCMerced    TIF-None ---> deleted (we do not have good models, as far as I remember)

p1=AID
p2=EuroSAT
p3=Gaofen
p4=Resisc45
p5=RSI-CB256





#p6=UCMerced ---> deleted

# -----------------------------------------------------------------------------
# PNG75 - PNG<zlib compression level><data encoding filter type> (PNG STANDARD)
# -----------------------------------------------------------------------------
#source ./png.sh $p1/reference $p1/ 75
source ./png.sh $p2/reference $p2/ 75
source ./png.sh $p3/reference $p3/ 75 # ---> Referance
#source ./png.sh $p4/reference $p4/ 75
source ./png.sh $p5/reference $p5/ 75
#source ./png.sh $p6/reference $p6/ 75

# -----------------------------------------------------------------------------
# PNG95
# -----------------------------------------------------------------------------
#source ./png.sh $p1/reference $p1/ 95
source ./png.sh $p2/reference $p2/ 95
source ./png.sh $p3/reference $p3/ 95
#source ./png.sh $p4/reference $p4/ 95
source ./png.sh $p5/reference $p5/ 95
#source ./png.sh $p6/reference $p6/ 95

# -----------------------------------------------------------------------------
# JPEG75 - JPEG<Quality Factor - 0-95> (JPEG STANDARD)
# -----------------------------------------------------------------------------
#source ./jpg.sh $p1/reference $p1/ 75 # ---> Referance
source ./jpg.sh $p2/reference $p2/ 75 # ---> Referance
source ./jpg.sh $p3/reference $p3/ 75
#source ./jpg.sh $p4/reference $p4/ 75 # ---> Referance
source ./jpg.sh $p5/reference $p5/ 75
#source ./jpg.sh $p6/reference $p6/ 75

# -----------------------------------------------------------------------------
# JPEG50
# -----------------------------------------------------------------------------
#source ./jpg.sh $p1/reference $p1/ 50
source ./jpg.sh $p2/reference $p2/ 50
source ./jpg.sh $p3/reference $p3/ 50
#source ./jpg.sh $p4/reference $p4/ 50
source ./jpg.sh $p5/reference $p5/ 50
#source ./jpg.sh $p6/reference $p6/ 50

# -----------------------------------------------------------------------------
# JPEG25
# -----------------------------------------------------------------------------
#source ./jpg.sh $p1/reference $p1/ 25
source ./jpg.sh $p2/reference $p2/ 25
source ./jpg.sh $p3/reference $p3/ 25
#source ./jpg.sh $p4/reference $p4/ 25
source ./jpg.sh $p5/reference $p5/ 25
#source ./jpg.sh $p6/reference $p6/ 25

# -----------------------------------------------------------------------------
# JPEG10
# -----------------------------------------------------------------------------
#source ./jpg.sh $p1/reference $p1/ 10
source ./jpg.sh $p2/reference $p2/ 10
source ./jpg.sh $p3/reference $p3/ 10
#source ./jpg.sh $p4/reference $p4/ 10
source ./jpg.sh $p5/reference $p5/ 10
#source ./jpg.sh $p6/reference $p6/ 10

# -----------------------------------------------------------------------------
# JPEG5
# -----------------------------------------------------------------------------
#source ./jpg.sh $p1/reference $p1/ 5
source ./jpg.sh $p2/reference $p2/ 5
source ./jpg.sh $p3/reference $p3/ 5
#source ./jpg.sh $p4/reference $p4/ 5
source ./jpg.sh $p5/reference $p5/ 5
#source ./jpg.sh $p6/reference $p6/ 5

# -----------------------------------------------------------------------------
# JPEG1
# -----------------------------------------------------------------------------
#source ./jpg.sh $p1/reference $p1/ 1
source ./jpg.sh $p2/reference $p2/ 1
source ./jpg.sh $p3/reference $p3/ 1
#source ./jpg.sh $p4/reference $p4/ 1
source ./jpg.sh $p5/reference $p5/ 1
#source ./jpg.sh $p6/reference $p6/ 1

# -----------------------------------------------------------------------------
# TIFF
# -----------------------------------------------------------------------------
s#ource ./tiff.sh $p1/reference $p1/
source ./tiff.sh $p2/reference $p2/
source ./tiff.sh $p3/reference $p3/
s#ource ./tiff.sh $p4/reference $p4/
#source ./tiff.sh $p5/reference $p5/ # ---> Referance
#s#ource ./tiff.sh $p6/reference $p6/ # ---> Referance

# -----------------------------------------------------------------------------
# BMP
# -----------------------------------------------------------------------------
##source ./bmp.sh $p1/reference $p1/ --> not used anymore
source ./bmp.sh $p2/reference $p2/
source ./bmp.sh $p3/reference $p3/
##source ./bmp.sh $p4/reference $p4/ --> not used anymore
source ./bmp.sh $p5/reference $p5/
##source ./bmp.sh $p6/reference $p6/ --> not used anymore







#source ./png.sh $1/reference $1/ 75
#source ./png.sh $1/reference $1/ 95
#
#source ./jpg.sh $1/reference $1/ 75
#source ./jpg.sh $1/reference $1/ 50
#source ./jpg.sh $1/reference $1/ 25
#source ./jpg.sh $1/reference $1/ 10
#source ./jpg.sh $1/reference $1/ 5
#source ./jpg.sh $1/reference $1/ 1
#
#source ./tiff.sh $1/reference $1/
#
#source ./bmp.sh $1/reference $1/

























