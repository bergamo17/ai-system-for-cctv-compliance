LOCAL_DIR := $(HOME)/Simulation/output/videos/violation
MOUNT_DIR := /mnt/c/Users/user/Documents/Footage/result
VENV := /mnt/c/Users/user/Documents/Simulation/venv/bin/python

run:
	$(VENV) main.py

sync:
	rsync -av --update $(LOCAL_DIR)/ $(MOUNT_DIR)/

run-sync: run sync

open:
	explorer.exe $$(wslpath -w $(MOUNT_DIR))

clean:
	find $(LOCAL_DIR) -type f -mtime +7 -delete

status:
	@echo "Local : $$(find $(LOCAL_DIR) -type f | wc -l) file"
	@echo "Mount : $$(find $(MOUNT_DIR) -type f | wc -l) file"

PHONY: run sync run-sync open clean status
