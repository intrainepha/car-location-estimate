package main

import (
	"encoding/xml"
	"flag"
	"log"
	"os"
	"path"
	"path/filepath"
	"strconv"
	"strings"
)

type XMLData struct {
	Size Size   `xml:"size"`
	Roi  ROI    `xml:"roi"`
	Obj  Object `xml:"obj"`
}

type Size struct {
	W float64 `xml:"w"`
	H float64 `xml:"h"`
}

type ROI struct {
	Size Size `xml:"size"`
	Box  Box  `xml:"bbox"`
}
type Box struct {
	Xmin float64 `xml:"xmin"`
	Ymin float64 `xml:"ymin"`
	Xmax float64 `xml:"xmax"`
	Ymax float64 `xml:"ymax"`
}
type Object struct {
	Name string   `xml:"name"`
	Box  Box      `xml:"bndbox"`
	Loc  Location `xml:"location"`
}

type Location struct {
	Y string `xml:"y"`
	X string `xml:"x"`
}

func crop() error {
	/*Crop Region of interest (ROI) from image with label formated in kitti approch:

	Args:
		root(string): Directory contains data files

	Returns:
		error
	*/

	return nil
}

func list(root string, cls []string) error {
	/*Generate paths.txt file, which contains data paths with each
	data path in one line, this file format is required by yolo-v3.

	Args:
		root(string): Directory contains data files
			dataset_directory/
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
	if err != nil {
		log.Fatal(err)
	}
	txt := path.Join(root, "paths.txt")
	buffer, err := os.OpenFile(txt, os.O_RDWR|os.O_CREATE, 0755)
	if err != nil {
		log.Fatal(err)
		return err
	}
	defer buffer.Close()
	for i, c := range cls {
		log.Println(i, c)
		imgDir := path.Join(root, c, "images")
		_, err := os.Stat(imgDir)
		if err != nil {
			log.Fatal(err)
			return err
		}
		imgs, err := os.ReadDir(imgDir)
		if err != nil {
			log.Fatal(err)
			return err
		}
		for _, f := range imgs {
			path := path.Join(imgDir, f.Name())
			buffer.WriteString(path + "\n")
		}

	}
	return nil
}

func genTxt(root string, cls []string) error {
	/*Read *.xml lable files and generate *.txt files.

	Args:
		root(string): Dataset directory
		cls([]string): classes you choose to generate

	Returns:
		error
	*/

	root, err := filepath.Abs(root)
	if err != nil {
		log.Fatal(err)
	}
	for id, c := range cls {
		xmlDir := path.Join(root, c, "annotations")
		txtDir := path.Join(root, c, "labels")
		xmls, err := os.ReadDir(xmlDir)
		if err != nil {
			log.Fatal(err)
			return err
		}
		for _, x := range xmls {
			xmlPath := path.Join(xmlDir, x.Name())
			xmlData, _ := os.ReadFile(xmlPath)
			var data XMLData
			_ = xml.Unmarshal([]byte(xmlData), &data)
			objXCtr := strconv.FormatFloat((data.Obj.Box.Xmin+data.Obj.Box.Xmax)/2/data.Roi.Size.W, 'g', -1, 64)
			objYCtr := strconv.FormatFloat((data.Obj.Box.Ymin+data.Obj.Box.Ymax)/2/data.Roi.Size.H, 'g', -1, 64)
			objW := strconv.FormatFloat((data.Obj.Box.Xmax-data.Obj.Box.Xmin+1)/data.Roi.Size.W, 'g', -1, 64)
			objH := strconv.FormatFloat((data.Obj.Box.Ymax-data.Obj.Box.Ymin+1)/data.Roi.Size.H, 'g', -1, 64)
			roiXCtr := strconv.FormatFloat((data.Roi.Box.Xmin+data.Roi.Box.Xmax)/2/data.Size.W, 'g', -1, 64)
			roiYCtr := strconv.FormatFloat((data.Roi.Box.Ymin+data.Roi.Box.Ymax)/2/data.Size.H, 'g', -1, 64)
			roiW := strconv.FormatFloat((data.Obj.Box.Xmax-data.Obj.Box.Xmin+1)/data.Size.W, 'g', -1, 64)
			roiH := strconv.FormatFloat((data.Obj.Box.Ymax-data.Obj.Box.Ymin+1)/data.Size.H, 'g', -1, 64)
			str := []string{
				strconv.Itoa(id),
				objXCtr, objYCtr, objW, objH,
				data.Obj.Loc.Y, data.Obj.Loc.X,
				roiXCtr, roiYCtr, roiW, roiH,
			}
			txtName := strings.Replace(x.Name(), ".xml", ".txt", 1)
			txtPath := path.Join(txtDir, txtName)
			buffer, err := os.OpenFile(txtPath, os.O_RDWR|os.O_CREATE, 0755)
			if err != nil {
				log.Fatal(err)
				return err
			}
			defer buffer.Close()
			buffer.WriteString(strings.Join(str, " "))
		}
	}

	return nil
}

func main() {
	listCmd := flag.NewFlagSet("list", flag.ExitOnError)
	listDir := listCmd.String("dir", "", "Directory")
	listCls := listCmd.String("cls", "", "Classes")
	gtCmd := flag.NewFlagSet("gt", flag.ExitOnError)
	gtDir := gtCmd.String("dir", "", "Directory")
	gtCls := gtCmd.String("cls", "", "Classes")
	if len(os.Args) < 2 {
		log.Fatal("Expected subcommands!")
		os.Exit(1)
	}
	switch os.Args[1] {
	case "list":
		listCmd.Parse(os.Args[2:])
		classes := strings.Split(*listCls, ",")
		err := list(*listDir, classes)
		if err != nil {
			log.Fatal(err)
		}
	case "gt":
		gtCmd.Parse(os.Args[2:])
		classes := strings.Split(*gtCls, ",")
		err := genTxt(*gtDir, classes)
		if err != nil {
			log.Fatal(err)
		}
	default:
		log.Println("Expected subcommands")
		os.Exit(1)
	}
}
