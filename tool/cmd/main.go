package main

import (
	"flag"
	"image/color"
	"log"
	"math/rand"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	op "github.com/intrainepha/car-location-estimation/tool/src/ops"
	tp "github.com/intrainepha/car-location-estimation/tool/src/tps"
)

/*
Transform data into ROI format:

	| Class_ID | Bounding_box | Location | ROI Bounding_box |

Args:

	id(int): class ID
	b(*tp.Box): object box that relative to ROI image
	l(*tp.Location): x-distence and Y-distance in the real world
	rb(*tp.Box): ROI box that relative to origin image

Returns:

	(string): Formed ROI label string
*/
func formROILabel(id int, b *tp.Box, l *tp.Location, rb *tp.Box) string {
	ss := []string{
		strconv.Itoa(id),
		op.F642Str(b.Scl.Xc), op.F642Str(b.Scl.Yc),
		op.F642Str(b.Scl.W), op.F642Str(b.Scl.H),
		op.F642Str(l.Y), op.F642Str(l.X),
		op.F642Str(rb.Scl.Xc), op.F642Str(rb.Scl.Yc),
		op.F642Str(rb.Scl.W), op.F642Str(rb.Scl.H),
	}
	return strings.Join(ss, " ")
}

/*
Crop Region of interest (ROI) from image with label formated in kitti approch.

	root/
		├──images
		|   └──*.png
		└──labels
		└──*.txt

Args:

	root(string): Directory contains data files
	freq(int): Frequence for filtering images

Returns:

	error
*/
func runCrop(root string, freq int, clsPath string) {

	root, err := filepath.Abs(root)
	op.CheckE(err)
	clsPath, err = filepath.Abs(clsPath)
	op.CheckE(err)
	clsF, err := tp.NewFile(clsPath)
	op.CheckE(err)
	cls := clsF.ReadLines()
	ds := path.Base(root)
	imDir := path.Join(root, "images")
	lbDir := path.Join(root, "labels")
	imDirROI := path.Join(strings.Replace(root, ds, ds+"_roi", 1), "images")
	lbDirROI := path.Join(strings.Replace(root, ds, ds+"_roi", 1), "labels")
	op.CleanDir(imDirROI, lbDirROI)
	files, err := os.ReadDir(imDir)
	op.CheckE(err)
	for i, f := range files {
		if i%freq != 0 {
			continue
		}
		im, err := tp.NewImData().Load(path.Join(imDir, f.Name()))
		op.CheckE(err)
		p := path.Join(lbDir, strings.Replace(f.Name(), ".png", ".txt", 1))
		oFile, err := tp.NewFile(p)
		op.CheckE(err)
		defer oFile.Close()
		op.CheckE(oFile.Read())
		for j, l := range oFile.ReadLines() {
			kt := tp.NewKITTI(cls, l)
			id := kt.Cls.GetID(kt.Name)
			if kt.FilterOut() {
				continue
			}
			rct := tp.NewRect(kt.Rct.Xtl, kt.Rct.Ytl, kt.Rct.Xbr, kt.Rct.Ybr)
			ob, rb, b, offset := kt.MakeROI(&im.Sz, rct, 0.25)
			imSub := im.Crop(&rb.Rct)
			//TODO: loading image by its suffix
			p = path.Join(imDirROI, strings.Replace(f.Name(), ".png", "_"+strconv.Itoa(j)+".jpg", 1))
			imSub.Save(p)
			p = path.Join(lbDirROI, strings.Replace(f.Name(), ".png", "_"+strconv.Itoa(j)+".txt", 1))
			tFile, err := tp.NewFile(p)
			op.CheckE(err)
			defer tFile.Close()
			err = tFile.WriteLine(formROILabel(id, b, &kt.Loc, rb))
			op.CheckE(err)
			orient := [4][4]float64{
				{0, 1, 0, 1},   // move up
				{0, -1, 0, -1}, // move down
				{-1, 0, -1, 0}, // move left
				{1, 0, 1, 0},   // move right
			}
			for k, m := range orient {
				rd := (400 + float64(rand.Intn(500))) / 1000 //random number in [0.4, 0.9]
				step := [4]float64{
					m[0] * float64(offset.X) * rd, m[1] * float64(offset.Y) * rd,
					m[2] * float64(offset.X) * rd, m[3] * float64(offset.Y) * rd,
				}
				rct := tp.NewRect(
					int(float64(ob.Rct.Xtl)+step[0]), int(float64(ob.Rct.Ytl)+step[1]),
					int(float64(ob.Rct.Xbr)+step[2]), int(float64(ob.Rct.Ybr)+step[3]),
				)
				_, rb, b, _ := kt.MakeROI(&im.Sz, rct, 0.25)
				imSub = im.Crop(&rb.Rct)
				p := path.Join(
					imDirROI,
					strings.Replace(f.Name(), ".png", "_"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+".jpg", 1),
				)
				imSub.Save(p)
				p = path.Join(
					lbDirROI,
					strings.Replace(f.Name(), ".png", "_"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+".txt", 1),
				)
				atFile, err := tp.NewFile(p)
				op.CheckE(err)
				defer atFile.Close()
				atFile.WriteLine(formROILabel(id, b, &kt.Loc, rb))
			}
		}
	}
}

/*
Draw bounding box of object to image

Args:

	root(string): Directory contains data files

Returns:

	error
*/
func runVis(root string) {
	root, err := filepath.Abs(root)
	op.CheckE(err)
	imDir := path.Join(root, "images")
	lbDir := path.Join(root, "labels")
	fs, err := os.ReadDir(lbDir)
	op.CheckE(err)
	for _, f := range fs {
		imPath := path.Join(imDir, strings.Replace(f.Name(), ".txt", ".jpg", 1))
		im := tp.NewImData()
		im.Load(imPath)
		lbPath := path.Join(imDir, f.Name())
		lb, err := tp.NewFile(lbPath)
		op.CheckE(err)
		lbs := strings.Split(lb.Content, " ")
		b := tp.NewBox(op.Str2int(lbs[0]), tp.NewRect(0, 0, 0, 0), tp.NewSize(0, 0))
		b.ImSz = im.Sz
		b.Scl = *tp.NewScl(
			op.Str2f64(lbs[1]), op.Str2f64(lbs[2]),
			op.Str2f64(lbs[3]), op.Str2f64(lbs[4]),
		)
		b.UnScale()
		im.DrawRect(&b.Rct, color.RGBA{
			A: 255,
			R: 255,
			G: 1,
			B: 1,
		})
		im.Save(strings.Replace(imPath, "images", "vis", 1))
	}

}

/*
Generate paths.txt file, which contains data paths with each
data path in one line, this file format is required by yolo-v3.

Args:

	root(string): Directory contains data files
		root/
			├──class_0
			|   ├──data_file_0
			|   ├── ...
			|   └──data_file_n
			├── ...
			└──class_n
				├──data_file_0
				├── ...
				└──data_file_n
	cls([]string): classes you choose to generate

Returns:

	error
*/
func runList(root string, cls []string) {
	root, err := filepath.Abs(root)
	op.CheckE(err)
	txt := path.Join(root, "paths.txt")
	file, err := os.OpenFile(txt, os.O_RDONLY|os.O_CREATE, 0755)
	op.CheckE(err)
	defer file.Close()
	for i, c := range cls {
		log.Println(i, c)
		imgDir := path.Join(root, c, "images")
		_, err := os.Stat(imgDir)
		op.CheckE(err)
		files, err := os.ReadDir(imgDir)
		op.CheckE(err)
		for _, f := range files {
			path := path.Join(imgDir, f.Name())
			file.WriteString(path + "\n")
		}
	}
}

func main() {
	listCmd := flag.NewFlagSet("list", flag.ExitOnError)
	listDir := listCmd.String("dir", "", "Directory")
	listCls := listCmd.String("cls", "", "Classes")

	cropCmd := flag.NewFlagSet("crop", flag.ExitOnError)
	cropDir := cropCmd.String("dir", "", "Directory")
	cropFreq := cropCmd.Int("freq", 1, "Frequence")
	cropCls := cropCmd.String("cls", "", "Classes file")

	visCmd := flag.NewFlagSet("vis", flag.ExitOnError)
	visDir := visCmd.String("dir", "", "Directory")

	if len(os.Args) < 2 {
		log.Fatal("Expected subcommands!")
	}

	switch os.Args[1] {
	case "list":
		listCmd.Parse(os.Args[2:])
		classes := strings.Split(*listCls, ",")
		runList(*listDir, classes)
	case "crop":
		cropCmd.Parse(os.Args[2:])
		runCrop(*cropDir, *cropFreq, *cropCls)
	case "vis":
		cropCmd.Parse(os.Args[2:])
		runVis(*visDir)
	default:
		log.Println("Expected subcommands")
		os.Exit(1)
	}
}
