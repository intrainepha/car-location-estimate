package file

import (
	"bytes"
	"image"
	"image/draw"
	"image/jpeg"
	_ "image/png"
	"log"
	"os"
	"strings"

	op "github.com/intrainepha/car-location-estimation/tool/src/ops"
	tp "github.com/intrainepha/car-location-estimation/tool/src/types"
)

func LoadImg(path string) (image.Image, *tp.Size) {
	/*Load image fron a file path

	Args:
		path(string): Path to image file

	Returns:
		(image.Image): Readed image data
		([2]int): Image size=[width, height]
	*/

	imgBytes, err := os.ReadFile(path)
	op.CheckE(err)
	imgInfo, _, err := image.DecodeConfig(bytes.NewReader(imgBytes))
	op.CheckE(err)
	size := tp.NewSize(float64(imgInfo.Width), float64(imgInfo.Height))
	data, _, err := image.Decode(bytes.NewReader(imgBytes))
	op.CheckE(err)

	return data, size
}

func SaveImg(path string, img image.Image) {
	/*Save image in format=[png, jpg, gif]

	Args:
		path(image.Image): save path
		img(image.Image): Readed image data

	Returns:
		None
	*/

	formats := [...]string{"png", "jpg"}
	strs := strings.Split(path, ".")
	suffix := strs[len(strs)-1]
	var confirmed bool = false
	for _, f := range formats {
		if suffix == f {
			confirmed = true
			break
		}
	}
	if !confirmed {
		log.Panic("Unsupported format:", suffix)
	}
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE, 0755)
	op.CheckE(err)
	err = jpeg.Encode(f, img, &jpeg.Options{Quality: 100})
	op.CheckE(err)
}

func CropImg(img image.Image, rect image.Rectangle) image.Image {
	/*Crop Image

	Args:
		img(image.Image): Readed image data
		rect(image.Image): REctangle=[Xtl, Ytl, Xbr, Ybr]

	Returns:
		(image.Image)
	*/

	var res image.Image
	type subImageSupported interface {
		SubImage(r image.Rectangle) image.Image
	}
	if sImg, ok := img.(subImageSupported); ok {
		res = sImg.SubImage(rect)
	} else {
		res := image.NewRGBA(rect)
		draw.Draw(res, rect, img, rect.Min, draw.Src)
	}

	return res
}
