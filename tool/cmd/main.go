package main

// TODO: Goroutine

import (
	"flag"
	"image/color"
	"io/fs"
	"log"
	"math/rand"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	op "github.com/intrainepha/car-location-estimation/tool/src/ops"
	tp "github.com/intrainepha/car-location-estimation/tool/src/tps"
)

/*
Transform data into ROI format:

	| Class_ID | Bounding_box | Location | ROI Bounding_box |

Args:

	id int: class ID
	b *tp.Box: object box that relative to ROI image
	l *tp.Location: x-distence and Y-distance in the real world
	rb *tp.Box: ROI box that relative to origin image

Returns:

	string: Formed ROI label string
*/
func formROILabel(id int, b *tp.Box, l *tp.Point[float64], rb *tp.Box) string {
	ss := []string{
		strconv.Itoa(id),
		op.Ftos(b.Scl.Xc), op.Ftos(b.Scl.Yc),
		op.Ftos(b.Scl.W), op.Ftos(b.Scl.H),
		op.Ftos(l.Y), op.Ftos(l.X),
		op.Ftos(rb.Scl.Xc), op.Ftos(rb.Scl.Yc),
		op.Ftos(rb.Scl.W), op.Ftos(rb.Scl.H),
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

	root string: Directory contains data files
	freq int: Frequence for filtering images

Returns:

	None
*/
func runCrop(root string, freq int, clsPath string) {
	defer op.Timer(time.Now(), "crop")
	clsF := tp.NewFile(clsPath)
	cls := clsF.ReadLines()
	ds := path.Base(root)
	imDir := path.Join(root, "images")
	lbDir := path.Join(root, "labels")
	imDirROI := path.Join(strings.Replace(root, ds, ds+"_roi", 1), "images")
	lbDirROI := path.Join(strings.Replace(root, ds, ds+"_roi", 1), "labels")
	op.CleanDir(imDirROI, lbDirROI)
	files, _ := os.ReadDir(imDir)
	steps := [4][4]float64{
		{0, 1, 0, 1},   // move up
		{0, -1, 0, -1}, // move down
		{-1, 0, -1, 0}, // move left
		{1, 0, 1, 0},   // move right
	}
	var wg sync.WaitGroup
	for i, f := range files {
		if i%freq != 0 {
			continue
		}
		wg.Add(1)
		go func(i int, f fs.DirEntry) {
			im := tp.NewImData().Load(path.Join(imDir, f.Name()))
			p := path.Join(lbDir, strings.Replace(f.Name(), ".png", ".txt", 1))
			sfx := strings.Split(f.Name(), ".")[len(strings.Split(f.Name(), "."))-1]
			oFile := tp.NewFile(p)
			defer oFile.Close()
			for j, l := range oFile.ReadLines() {
				kt := tp.NewKITTI(cls, l)
				id := kt.Cls.GetID(kt.Name)
				if kt.FilterOut() {
					continue
				}
				rct := tp.NewRect(kt.Rct.Xtl, kt.Rct.Ytl, kt.Rct.Xbr, kt.Rct.Ybr)
				ob, rb, b, ofs := kt.MakeROI(&im.Sz, rct, [4]float64{}, 0.25)
				imSub := im.Crop(&rb.Rct)
				p = path.Join(imDirROI, strings.Replace(f.Name(), "."+sfx, "_"+strconv.Itoa(j)+".jpg", 1))
				imSub.Save(p)
				p = path.Join(lbDirROI, strings.Replace(f.Name(), "."+sfx, "_"+strconv.Itoa(j)+".txt", 1))
				tFile := tp.NewFile(p)
				defer tFile.Close()
				tFile.WriteLine(formROILabel(id, b, &kt.Loc, rb))
				for k, s := range steps {
					rd := (400 + float64(rand.Intn(500))) / 1000 //random number in 0.4~0.9
					trans := [4]float64{
						s[0] * float64(ofs.X) * rd, s[1] * float64(ofs.Y) * rd,
						s[2] * float64(ofs.X) * rd, s[3] * float64(ofs.Y) * rd,
					}
					_, rb, b, _ := kt.MakeROI(&im.Sz, &ob.Rct, trans, 0.25)
					imSub := im.Crop(&rb.Rct)
					p := path.Join(
						imDirROI,
						strings.Replace(f.Name(), "."+sfx, "_"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+".jpg", 1),
					)
					imSub.Save(p)
					p = path.Join(
						lbDirROI,
						strings.Replace(f.Name(), "."+sfx, "_"+strconv.Itoa(j)+"_"+strconv.Itoa(k)+".txt", 1),
					)
					atFile := tp.NewFile(p)
					defer atFile.Close()
					atFile.WriteLine(formROILabel(id, b, &kt.Loc, rb))
				}
			}
			wg.Done()
		}(i, f)
	}
	wg.Wait()
}

/*
Draw bounding box of object to image

Args:

	root string: Directory contains data files

Returns:

	None
*/
func runVis(root string) {
	defer op.Timer(time.Now(), "vis")
	imDir := path.Join(root, "images")
	lbDir := path.Join(root, "labels")
	visDir := path.Join(root, "vis")
	op.CleanDir(visDir)
	fs, err := os.ReadDir(imDir)
	if err != nil {
		log.Panic(err)
	}
	var wg sync.WaitGroup
	for _, f := range fs {
		fname := f.Name()
		wg.Add(1)
		go func(f string) {
			imPath := path.Join(imDir, fname)
			im := tp.NewImData()
			im.Load(imPath)
			sfx := strings.Split(fname, ".")[len(strings.Split(fname, "."))-1]
			lbPath := path.Join(lbDir, strings.Replace(fname, "."+sfx, ".txt", 1))
			lb := tp.NewFile(lbPath).Read()
			lbs := strings.Split(lb, " ")
			b := tp.NewBox(op.Stoi(lbs[0]), tp.NewRect(0, 0, 0, 0), tp.NewSize(0, 0))
			b.ImSz = im.Sz
			b.Scl = *tp.NewScl(
				op.Stof(lbs[1]), op.Stof(lbs[2]),
				op.Stof(lbs[3]), op.Stof(lbs[4]),
			)
			b.UnScale()
			im.DrawRect(&b.Rct, color.RGBA{A: 255, R: 0, G: 255, B: 0})
			im.Save(path.Join(visDir, fname))
			wg.Done()
		}(fname)
	}
	wg.Wait()
}

/*
Generate paths.txt file, which contains data paths with each
data path in one line, this file format is required by yolo-v3.

Args:

	root string: Directory contains data files
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
	cls []string: classes you choose to generate

Returns:

	None
*/
func runList(root string, cls []string) {
	defer op.Timer(time.Now(), "list")
	txt := path.Join(root, "paths.txt")
	file, err := os.OpenFile(txt, os.O_RDONLY|os.O_CREATE, 0755)
	if err != nil {
		log.Panic(err)
	}
	defer file.Close()
	var wg sync.WaitGroup
	for i, c := range cls {
		wg.Add(1)
		go func(i int, c string) {
			imgDir := path.Join(root, c, "images")
			_, _ = os.Stat(imgDir)
			files, _ := os.ReadDir(imgDir)
			for _, f := range files {
				path := path.Join(imgDir, f.Name())
				file.WriteString(path + "\n")
			}
			wg.Done()
		}(i, c)
	}
	wg.Wait()
}

func main() {
	cropCmd := flag.NewFlagSet("crop", flag.ExitOnError)
	cropDir := cropCmd.String("dir", "", "Directory")
	cropFreq := cropCmd.Int("freq", 1, "Frequence")
	cropCls := cropCmd.String("cls", "", "Classes file")

	visCmd := flag.NewFlagSet("vis", flag.ExitOnError)
	visDir := visCmd.String("dir", "", "Directory")

	listCmd := flag.NewFlagSet("list", flag.ExitOnError)
	listDir := listCmd.String("dir", "", "Directory")
	listCls := listCmd.String("cls", "", "Classes")

	if len(os.Args) < 2 {
		log.Fatal("Expected subcommands!")
	}

	switch os.Args[1] {
	case "list":
		listCmd.Parse(os.Args[2:])
		r, _ := filepath.Abs(*listDir)
		classes := strings.Split(*listCls, ",")
		runList(r, classes)
	case "crop":
		cropCmd.Parse(os.Args[2:])
		r, _ := filepath.Abs(*cropDir)
		clsPath, _ := filepath.Abs(*cropCls)
		runCrop(r, *cropFreq, clsPath)
	case "vis":
		visCmd.Parse(os.Args[2:])
		r, _ := filepath.Abs(*visDir)
		runVis(r)
	default:
		log.Panic("Expected subcommands")
		os.Exit(1)
	}
}
