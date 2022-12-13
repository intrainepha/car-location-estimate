package ops

import "os"

func CleanDir(dir string) {
	err := os.RemoveAll(dir)
	CheckE(err)
	err = os.MkdirAll(dir, 0755)
	CheckE(err)
}
