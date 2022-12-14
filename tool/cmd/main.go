package main

import (
	"flag"
	"log"
	"math/rand"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	pb "github.com/schollz/progressbar/v3"

	op "github.com/intrainepha/car-location-estimation/tool/src/ops"
	tp "github.com/intrainepha/car-location-estimation/tool/src/tps"
)

func runCrop(root string, freq int, cls []string) error {
	/*Crop Region of interest (ROI) from image with label formated in kitti approch.
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

	root, err := filepath.Abs(root)
	op.CheckE(err)
	imDir := path.Join(root, "images")
	lbDir := path.Join(root, "labels")
	imDirROI := path.Join(root, "roi_img")
	op.CleanDir(imDirROI)
	lbDirROI := path.Join(root, "roi_label")
	op.CleanDir(lbDirROI)
	files, err := os.ReadDir(imDir)
	op.CheckE(err)
	bar := pb.Default(int64(len(files)), "Processing files:")
	for i, f := range files {
		bar.Add(1)
		if i%freq != 0 {
			continue
		}
		im := tp.NewIm()
		err := im.Load(path.Join(imDir, f.Name()))
		op.CheckE(err)
		p := path.Join(lbDir, strings.Replace(f.Name(), ".png", ".txt", 1))
		txt, err := tp.NewTXT(p)
		op.CheckE(err)
		op.CheckE(txt.Load())
		for j, l := range txt.ReadLines() {
			kt := tp.NewKITTI(l)
			clsID, err := kt.Check(cls)
			if err != nil {
				continue
			}
			rct := tp.NewRect(kt.Rct.Xtl, kt.Rct.Ytl, kt.Rct.Xbr, kt.Rct.Ybr)
			ob, rb, b, offset := kt.MakeROI(&im.Sz, rct, 0.25)
			imSub := im.Crop(&rb.Rct)
			p = path.Join(imDirROI, strings.Replace(f.Name(), ".png", "_"+strconv.Itoa(j)+".jpg", 1))
			imSub.Save(p)
			p = path.Join(lbDirROI, strings.Replace(f.Name(), ".png", "_"+strconv.Itoa(j)+".txt", 1))
			tp.SaveTXT(p, clsID, b, &kt.Loc, rb)
			// Augment
			orient := [4][4]float64{
				{0, 1, 0, 1},   // move up
				{0, -1, 0, -1}, // move down
				{-1, 0, -1, 0}, // move left
				{1, 0, 1, 0},   // move right
			}
			for k, m := range orient {
				rd := (400 + float64(rand.Intn(500))) / 1000 //random number in [0.4, 0.9]
				step := [4]float64{
					m[0] * offset.X * rd, m[1] * offset.Y * rd,
					m[2] * offset.X * rd, m[3] * offset.Y * rd,
				}
				rct := tp.NewRect(
					ob.Rct.Xtl+step[0],
					ob.Rct.Ytl+step[1],
					ob.Rct.Xbr+step[2],
					ob.Rct.Ybr+step[3],
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
					strings.Replace(f.Name(), ".png", "_"+strconv.Itoa(j)+"_"+".txt", 1),
				)
				tp.SaveTXT(p, clsID, b, &kt.Loc, rb)
			}
		}
	}

	return nil
}

func runList(root string, cls []string) error {
	/*Generate paths.txt file, which contains data paths with each
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

	root, err := filepath.Abs(root)
	log.Println("Target Directory:", root)
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
	return nil
}

func main() {
	listCmd := flag.NewFlagSet("list", flag.ExitOnError)
	listDir := listCmd.String("dir", "", "Directory")
	listCls := listCmd.String("cls", "", "Classes")

	cropCmd := flag.NewFlagSet("crop", flag.ExitOnError)
	cropDir := cropCmd.String("dir", "", "Directory")
	cropFreq := cropCmd.Int("freq", 1, "Frequence")
	cropCls := cropCmd.String("cls", "", "Classes")

	if len(os.Args) < 2 {
		log.Fatal("Expected subcommands!")
	}

	switch os.Args[1] {
	case "list":
		listCmd.Parse(os.Args[2:])
		classes := strings.Split(*listCls, ",")
		err := runList(*listDir, classes)
		op.CheckE(err)
	case "crop":
		cropCmd.Parse(os.Args[2:])
		classes := strings.Split(*cropCls, ",")
		err := runCrop(*cropDir, *cropFreq, classes)
		op.CheckE(err)
	default:
		log.Println("Expected subcommands")
		os.Exit(1)
	}
}
