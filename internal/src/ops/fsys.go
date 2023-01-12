package ops

import (
	"log"
	"os"
)

/*
Check if a directory exists

Args:

	d string: passed in directory

Returns:

	bool: status
*/
func Exists(d string) bool {
	_, err := os.Stat(d)
	return !os.IsNotExist(err)
}

/*
Clean up directories

Args:

	ds ...string: passed in directories

Returns:

	None
*/
func CleanDir(ds ...string) {
	for _, d := range ds {
		err := os.RemoveAll(d)
		if err != nil {
			log.Panic(err)
		}
		err = os.MkdirAll(d, 0755)
		if err != nil {
			log.Panic(err)
		}
	}
}
