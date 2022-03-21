OBJECT_FILE = "object_file"
PIC_OBJECT_FILE = "pic_object_file"
RDC_OBJECT_FILE = "rdc_object_file"
RDC_PIC_OBJECT_FILE = "rdc_pic_object_file"

ARCHIVE = "archive"
PIC_ARCHIVE = "pic_archive"

ARTIFACT_CATEGORIES = struct(
    object_file = OBJECT_FILE,
    pic_object_file = PIC_OBJECT_FILE,
    rdc_object_file = RDC_OBJECT_FILE,
    rdc_pic_object_file = RDC_PIC_OBJECT_FILE,
    archive = ARCHIVE,
    pic_archive = PIC_ARCHIVE,
)
