.PHONY: clean
clean:
	@find . -type f -exec grep -IL . "{}" \; | xargs rm -f

.PHONY: style
style:
	@gofumpt -l -w ./..
