package tps

import (
	"bytes"
	"image"
	"image/draw"
	"image/jpeg"
	_ "image/png"
	"log"
	"os"
	"strings"
)

type Size struct {
	W float64
	H float64
}

type Im struct {
	Data image.Image
	Sz   Size
}

func NewIm() *Im {
	return &Im{}
}

func (i *Im) Load(p string) error {
	/*Load image fron a file path

	Args:
		p(string): Path to image file

	Returns:
		(error): Error
	*/

	byte, err := os.ReadFile(p)
	if err != nil {
		return err
	}
	info, _, err := image.DecodeConfig(bytes.NewReader(byte))
	if err != nil {
		return err
	}
	i.Sz = *NewSize(float64(info.Width), float64(info.Height))
	data, _, err := image.Decode(bytes.NewReader(byte))
	if err != nil {
		return err
	}
	i.Data = data

	return nil
}

func (i *Im) Save(p string) error {
	/*Save image in format=[png, jpg]

	Args:
		p(image.Image): save path

	Returns:
		None
	*/

	formats := [...]string{"png", "jpg"}
	strs := strings.Split(p, ".")
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
	f, err := os.OpenFile(p, os.O_WRONLY|os.O_CREATE, 0755)
	if err != nil {
		return err
	}
	err = jpeg.Encode(f, i.Data, &jpeg.Options{Quality: 100})
	if err != nil {
		return err
	}

	return nil
}

func (i *Im) Crop(r *Rect) *Im {
	/*Crop Image

	Args:
		img(image.Image): Readed image data
		rect(image.Rectangle): Rectangle=[
			Min(Point)=[X, Y],
			Max(Point)=[X, Y]
		]

	Returns:
		(image.Image)
	*/

	var res image.Image
	ir := image.Rect(int(r.Xtl), int(r.Ytl), int(r.Xbr), int(r.Ybr))
	type subImageSupported interface {
		SubImage(r image.Rectangle) image.Image
	}
	if sImg, ok := i.Data.(subImageSupported); ok {
		res = sImg.SubImage(ir)
	} else {
		res := image.NewRGBA(ir)
		draw.Draw(res, ir, i.Data, ir.Min, draw.Src)
	}
	sz := *NewSize(r.Xbr-r.Xtl+1, r.Ybr-r.Ytl+1)

	return &Im{Data: res, Sz: sz}
}

func DrawRect() {

}
