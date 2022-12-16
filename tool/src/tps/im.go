package tps

import (
	"bytes"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	_ "image/png"
	"log"
	"math"
	"os"
	"strings"
)

type Size struct {
	W float64
	H float64
}

type ImData struct {
	*image.RGBA
	image.Image
	Sz Size
}

/*
Init type ImData

Args:

	None

Returns:

	 type ImData struct {
		*image.RGBA
		image.Image
		Sz Size
	}
*/
func NewImData() *ImData {
	return &ImData{}
}

/*
Load ImData fron a file path

Args:

	string: path to image file

Returns:

	error
*/
func (i *ImData) Load(p string) (*ImData, error) {
	bt, err := os.ReadFile(p)
	if err != nil {
		return &ImData{}, err
	}
	info, _, err := image.DecodeConfig(bytes.NewReader(bt))
	if err != nil {
		return &ImData{}, err
	}
	i.Sz = *NewSize(float64(info.Width), float64(info.Height))
	im, _, err := image.Decode(bytes.NewReader(bt))
	if err != nil {
		return &ImData{}, err
	}
	// bounds := im.Bounds()
	// rgba := image.NewRGBA(bounds)
	// draw.Draw(rgba, bounds, im, bounds.Min, draw.Over)
	// i.Im = rgba
	i.Image = im
	i.ToRGBA()

	return i, nil
}

/*
Transfor image.Image to image.RGBA

Args:

	None

Returns:

	None
*/
func (i *ImData) ToRGBA() {
	bounds := i.Image.Bounds()
	rgba := image.NewRGBA(bounds)
	draw.Draw(rgba, bounds, i.Image, bounds.Min, draw.Over)
	i.RGBA = rgba
}

/*
Save image in format=[png, jpg]

Args:

	p: save path

Returns:

	error
*/
func (i *ImData) Save(p string) error {
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
	err = jpeg.Encode(f, i.RGBA, &jpeg.Options{Quality: 100})
	if err != nil {
		return err
	}

	return nil
}

/*
Crop image

Args:

	*Rect:
		Xtl(float64): x of top-left point
		Ytl(float64): y of top-left point
		Xbr(float64): x of bottom-right point
		Ybr(float64): y of bottom-right point

Returns:

	*Image
*/
func (i *ImData) Crop(r *Rect) *ImData {
	ir := image.Rect(int(r.Xtl), int(r.Ytl), int(r.Xbr), int(r.Ybr))
	// var dst image.Image
	// type subImageSupported interface {
	// 	SubImage(r image.Rectangle) image.Image
	// }
	// if sImg, ok := i.Image.(subImageSupported); ok {
	// 	im := i.Image.SubImage(ir)
	// } else {
	// dst := image.NewRGBA(ir)
	// draw.Draw(dst, ir, i.Image, ir.Min, draw.Src)
	// }
	imD := NewImData()
	imD.Image = i.RGBA.SubImage(ir)
	imD.Sz = *NewSize(r.Xbr-r.Xtl+1, r.Ybr-r.Ytl+1)
	imD.ToRGBA()

	return imD
}

func (i *ImData) DrawRect(r *Rect, c color.Color) {
	shorter := math.Min(r.Xbr-r.Xtl, r.Ybr-r.Ytl)
	bold := int(shorter / 200)
	if bold < 1 {
		bold = 1
	}
	or := image.Rect(int(r.Xtl), int(r.Ytl), int(r.Xbr), int(r.Ybr))
	ir := image.Rect(
		int(r.Xtl)+bold, int(r.Ytl)+bold,
		int(r.Xbr)-bold, int(r.Ybr)-bold,
	)
	// bounds := i.Image.Bounds()
	// dst := image.NewRGBA(bounds)
	// draw.Draw(dst, bounds, i.Image, bounds.Min, draw.Over)
	imSub := i.Crop(r)
	for y := 0; y < int(imSub.Sz.H); y++ {
		for x := 0; x < int(imSub.Sz.W); x++ {
			if (y >= ir.Min.Y && y <= ir.Max.Y) || (x >= ir.Min.X && x <= ir.Max.X) {
				continue
			}
			imSub.RGBA.Set(x, y, c)
		}
	}
	draw.Draw(i.RGBA, or.Bounds(), imSub.RGBA, or.Min, draw.Over)
}
