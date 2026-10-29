//! Domain-based sectors (Experiment 5):
//!
//! Instead of partitioning tokens by their local syntactic category (which switches on
//! nearly every token and causes high LRU cache churn), files are grouped into 13 domain
//! sectors (00..12) according to their imports (`numpy`/`scipy`, `torch`/`tensorflow`,
//! `django`/`flask`, `unittest`/`pytest`, standard library categories, etc.).
//!
//! In the structured token `SS R C IIIIII`:
//! - `SS` (00..12) is the import domain group (the SECTOR loaded on demand).
//! - `R` (1 letter) encodes the syntactic role from the AST (`K` keyword, `D` defined name,
//!   `R` used name/reference, `N` numeric, `S` string, `O` operator, `P` punctuation/delimiter,
//!   `M` comment, `A` decorator, `I` import, `B` builtin, `F` format, `T` control).
//! - `C` (1 letter) encodes the syntactic condition (`G` global, `F` function, `C` class,
//!   `A` args, `E` expr, `S` spacing/format).
//! - `IIIIII` is the vocabulary ID within that domain sector.

use serde::{Deserialize, Serialize};
use crate::sector::parser::ParsedToken;
use crate::sector::token::StructuredToken;
use crate::sector::SectorKind;

pub const NUM_DOMAIN_SECTORS: usize = 13;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum DomainGroup {
    ScientificNumeric = 0,
    MachineLearning = 1,
    WebBackend = 2,
    NetworkingCloud = 3,
    TestingQa = 4,
    DataStorageFormats = 5,
    SystemOsCli = 6,
    RuntimeInspectPkg = 7,
    AsyncConcurrency = 8,
    GuiGraphicsMedia = 9,
    TextCryptoMath = 10,
    StdlibCoreTypes = 11,
    GeneralLocal = 12,
}

impl DomainGroup {
    pub fn id(&self) -> u8 {
        *self as u8
    }

    pub fn from_id(id: u8) -> Option<Self> {
        match id {
            0 => Some(Self::ScientificNumeric),
            1 => Some(Self::MachineLearning),
            2 => Some(Self::WebBackend),
            3 => Some(Self::NetworkingCloud),
            4 => Some(Self::TestingQa),
            5 => Some(Self::DataStorageFormats),
            6 => Some(Self::SystemOsCli),
            7 => Some(Self::RuntimeInspectPkg),
            8 => Some(Self::AsyncConcurrency),
            9 => Some(Self::GuiGraphicsMedia),
            10 => Some(Self::TextCryptoMath),
            11 => Some(Self::StdlibCoreTypes),
            12 => Some(Self::GeneralLocal),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::ScientificNumeric => "00_ScientificNumeric (numpy/scipy/pandas)",
            Self::MachineLearning => "01_MachineLearning (torch/tf/sklearn)",
            Self::WebBackend => "02_WebBackend (django/flask/web)",
            Self::NetworkingCloud => "03_NetworkingCloud (requests/http/cloud)",
            Self::TestingQa => "04_TestingQa (unittest/pytest/mock)",
            Self::DataStorageFormats => "05_DataStorage (json/sql/xml/archive)",
            Self::SystemOsCli => "06_SystemOsCli (os/sys/subprocess/cli)",
            Self::RuntimeInspectPkg => "07_RuntimeInspect (logging/inspect/ast)",
            Self::AsyncConcurrency => "08_AsyncConcurrency (threading/asyncio)",
            Self::GuiGraphicsMedia => "09_GuiGraphicsMedia (qt/tk/pil/cv2)",
            Self::TextCryptoMath => "10_TextCryptoMath (re/hashlib/math/random)",
            Self::StdlibCoreTypes => "11_StdlibCoreTypes (collections/typing/time)",
            Self::GeneralLocal => "12_GeneralLocal (local/other)",
        }
    }

    /// Specialized ecosystem domains carry higher specificity weight than ubiquitous stdlib helpers.
    fn specificity_weight(&self) -> u32 {
        match self {
            Self::ScientificNumeric
            | Self::MachineLearning
            | Self::WebBackend
            | Self::NetworkingCloud
            | Self::TestingQa
            | Self::DataStorageFormats
            | Self::AsyncConcurrency
            | Self::GuiGraphicsMedia => 3,
            Self::SystemOsCli
            | Self::RuntimeInspectPkg
            | Self::TextCryptoMath
            | Self::StdlibCoreTypes => 1,
            Self::GeneralLocal => 0,
        }
    }
}

/// Maps an identifier appearing inside an `import` / `from ... import` statement to a domain group.
pub fn classify_import_identifier(ident: &str) -> Option<DomainGroup> {
    match ident {
        // 00: Scientific & Numeric
        "numpy" | "np" | "scipy" | "pandas" | "pd" | "matplotlib" | "plt" | "pylab"
        | "seaborn" | "sns" | "sympy" | "statsmodels" | "h5py" | "tables" | "netCDF4"
        | "xarray" | "numba" | "cython" | "astropy" | "biopython" | "Bio" | "networkx" => {
            Some(DomainGroup::ScientificNumeric)
        }

        // 01: Machine Learning & AI
        "torch" | "torchvision" | "torchaudio" | "tensorflow" | "tf" | "keras" | "sklearn"
        | "xgboost" | "lightgbm" | "catboost" | "caffe" | "theano" | "lasagne" | "chainer"
        | "onnx" | "jax" | "flax" | "optuna" | "gym" | "nltk" | "spacy" | "gensim"
        | "transformers" | "datasets" => Some(DomainGroup::MachineLearning),

        // 02: Web Frameworks & Backend
        "django" | "flask" | "fastapi" | "tornado" | "bottle" | "pyramid" | "werkzeug"
        | "jinja2" | "mako" | "aiohttp" | "starlette" | "sanic" | "falcon" | "cherrypy"
        | "wtforms" | "celery" | "kombu" | "wsgi" | "wsgiref" | "cgi" | "cgitb" => {
            Some(DomainGroup::WebBackend)
        }

        // 03: Networking, HTTP & Cloud
        "requests" | "urllib" | "urllib2" | "urllib3" | "httplib" | "httplib2" | "http"
        | "socket" | "SocketServer" | "socketserver" | "ftplib" | "smtplib" | "poplib"
        | "imaplib" | "telnetlib" | "paramiko" | "fabric" | "twisted" | "boto" | "boto3"
        | "botocore" | "google" | "azure" | "grpc" | "zmq" | "websocket" | "xmlrpclib"
        | "xmlrpc" | "urlparse" | "Cookie" | "cookielib" => Some(DomainGroup::NetworkingCloud),

        // 04: Testing, QA & Debugging
        "unittest" | "unittest2" | "pytest" | "nose" | "mock" | "hypothesis" | "doctest"
        | "tox" | "coverage" | "faker" | "factory" | "pdb" | "bdb" | "trace" | "timeit"
        | "cProfile" | "profile" | "pstats" => Some(DomainGroup::TestingQa),

        // 05: Data Serialization, Storage, Databases & Archives
        "json" | "simplejson" | "ujson" | "pickle" | "cPickle" | "shelve" | "marshal"
        | "dbm" | "gdbm" | "csv" | "yaml" | "toml" | "configparser" | "ConfigParser"
        | "xml" | "ElementTree" | "cElementTree" | "lxml" | "bs4" | "BeautifulSoup"
        | "html" | "HTMLParser" | "sqlite3" | "sqlalchemy" | "pymongo" | "bson" | "redis"
        | "psycopg2" | "MySQLdb" | "pymysql" | "struct" | "base64" | "binascii" | "quopri"
        | "uu" | "zlib" | "gzip" | "bz2" | "lzma" | "zipfile" | "tarfile" => {
            Some(DomainGroup::DataStorageFormats)
        }

        // 06: System, OS, Filesystem & CLI
        "os" | "sys" | "subprocess" | "shutil" | "pathlib" | "posixpath" | "ntpath"
        | "glob" | "fnmatch" | "linecache" | "tempfile" | "signal" | "platform" | "pwd"
        | "grp" | "spwd" | "stat" | "errno" | "fcntl" | "resource" | "ctypes" | "mmap"
        | "io" | "StringIO" | "cStringIO" | "argparse" | "optparse" | "getopt" | "shlex"
        | "getpass" | "tty" | "pty" | "termios" | "curses" | "cmd" | "readline" | "rlcompleter" => {
            Some(DomainGroup::SystemOsCli)
        }

        // 07: Runtime, Logging, AST, Inspection & Packaging
        "logging" | "warnings" | "traceback" | "atexit" | "gc" | "weakref" | "inspect"
        | "dis" | "ast" | "symtable" | "tokenize" | "token" | "keyword" | "parser" | "symbol"
        | "imp" | "importlib" | "zipimport" | "pkgutil" | "modulefinder" | "runpy"
        | "pkg_resources" | "setuptools" | "distutils" | "pip" | "venv" | "site"
        | "sysconfig" | "code" | "codeop" | "compileall" | "py_compile" => {
            Some(DomainGroup::RuntimeInspectPkg)
        }

        // 08: Async, Threading & Concurrency
        "threading" | "thread" | "_thread" | "dummy_thread" | "multiprocessing" | "asyncio"
        | "concurrent" | "futures" | "queue" | "Queue" | "sched" | "select" | "selectors"
        | "asyncore" | "asynchat" | "gevent" | "eventlet" | "greenlet" => {
            Some(DomainGroup::AsyncConcurrency)
        }

        // 09: GUI, Graphics, Images & Multimedia
        "tkinter" | "Tkinter" | "ttk" | "tkMessageBox" | "tkFileDialog" | "tkSimpleDialog"
        | "tkColorChooser" | "tkFont" | "ScrolledText" | "Tix" | "turtle" | "PyQt4"
        | "PyQt5" | "PySide" | "PySide2" | "wx" | "gtk" | "gi" | "pygtk" | "gobject"
        | "PIL" | "Image" | "ImageDraw" | "ImageFont" | "cv2" | "imageio" | "skimage"
        | "pygame" | "pyglet" | "OpenGL" | "vtk" | "bpy" | "wave" | "chunk" | "audioop"
        | "aifc" | "sunau" | "sndhdr" | "imghdr" | "colorsys" => {
            Some(DomainGroup::GuiGraphicsMedia)
        }

        // 10: Text Processing, Regex, Cryptography & Math
        "re" | "sre_compile" | "sre_parse" | "sre_constants" | "string" | "unicodedata"
        | "difflib" | "textwrap" | "codecs" | "locale" | "gettext" | "hashlib" | "hmac"
        | "md5" | "sha" | "secrets" | "cryptography" | "Crypto" | "ssl" | "random"
        | "math" | "cmath" | "decimal" | "fractions" | "numbers" | "statistics" => {
            Some(DomainGroup::TextCryptoMath)
        }

        // 11: Stdlib Core Types, Collections, Functional & Time
        "collections" | "itertools" | "functools" | "operator" | "typing" | "types" | "abc"
        | "enum" | "dataclasses" | "copy" | "copy_reg" | "copyreg" | "pprint" | "reprlib"
        | "heapq" | "bisect" | "array" | "contextlib" | "time" | "datetime" | "calendar"
        | "pytz" | "dateutil" | "six" | "__future__" | "builtins" | "__builtin__"
        | "exceptions" | "UserDict" | "UserList" | "UserString" => {
            Some(DomainGroup::StdlibCoreTypes)
        }

        _ => None,
    }
}

/// Assigns a file (represented by its AST-parsed tokens) to one of the 13 domain sectors
/// according to its `Import` tokens.
pub fn classify_file_domain(tokens: &[ParsedToken]) -> DomainGroup {
    let mut scores = [0u32; NUM_DOMAIN_SECTORS];
    for t in tokens {
        if t.sector == SectorKind::Import {
            if let Some(dom) = classify_import_identifier(&t.text) {
                scores[dom.id() as usize] += dom.specificity_weight();
            }
        }
    }

    let mut best_dom = DomainGroup::GeneralLocal;
    let mut best_score = 0u32;
    for id in 0..NUM_DOMAIN_SECTORS - 1 {
        if scores[id] > best_score {
            best_score = scores[id];
            best_dom = DomainGroup::from_id(id as u8).unwrap();
        }
    }
    best_dom
}

/// Maps the original syntactic sector + definition/reference flags to the 1-character `rule`
/// field of the structured token `SS R C IIIIII`.
pub fn syntactic_role_code(sector: SectorKind, is_def: bool, is_ref: bool) -> char {
    if is_def || sector == SectorKind::DefinedName {
        return 'D';
    }
    if is_ref || sector == SectorKind::UsedName {
        return 'R';
    }
    match sector {
        SectorKind::Keyword => 'K',
        SectorKind::NumericLiteral => 'N',
        SectorKind::StringLiteral => 'S',
        SectorKind::Operator => 'O',
        SectorKind::Delimiter => 'P',
        SectorKind::Comment => 'M',
        SectorKind::Decorator => 'A',
        SectorKind::Import => 'I',
        SectorKind::BuiltinType => 'B',
        SectorKind::Format => 'F',
        SectorKind::TextPiece => 'T',
        SectorKind::Control | SectorKind::Symbol => 'T',
        SectorKind::DefinedName => 'D',
        SectorKind::UsedName => 'R',
    }
}

/// Remaps a file's syntax tokens so that `tok.sector` is the file's `DomainGroup` (00..12),
/// while `tok.rule` and `tok.condition` carry the syntactic role and AST scope. Dedicated
/// TextPiece tokens keep their shared sector so comment/string BPE is common across domains.
pub fn remap_tokens_to_domain(tokens: &[ParsedToken]) -> (DomainGroup, Vec<ParsedToken>) {
    let domain = classify_file_domain(tokens);
    let dom_sector = SectorKind::from_id(domain.id()).expect("domain id in 0..13");

    let remapped = tokens
        .iter()
        .map(|t| {
            let rule = syntactic_role_code(t.sector, t.is_definition, t.is_reference);
            let condition = if t.sector == SectorKind::Format {
                'S'
            } else {
                t.condition
            };
            ParsedToken {
                text: t.text.clone(),
                sector: if t.sector == SectorKind::TextPiece { SectorKind::TextPiece } else { dom_sector },
                rule,
                condition,
                is_definition: t.is_definition,
                is_reference: t.is_reference,
                byte_range: t.byte_range,
            }
        })
        .collect();

    (domain, remapped)
}

/// Builds the canonical `SS R C IIIIII` structured token for a domain-remapped token.
pub fn to_structured_domain_token(tok: &ParsedToken, id_in_sector: u32) -> StructuredToken {
    StructuredToken::new(tok.sector.id(), tok.rule, tok.condition, id_in_sector)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sector::parser::PythonAstParser;

    #[test]
    fn test_domain_classification_and_structured_token_encoding() {
        let mut parser = PythonAstParser::new();

        let ml_code = "import os\nimport torch\nimport torch.nn as nn\n\ndef forward(x):\n    return x + 1\n";
        let toks = parser.parse_code_with_format(ml_code).unwrap();
        let (dom, remapped) = remap_tokens_to_domain(&toks);
        assert_eq!(dom, DomainGroup::MachineLearning);

        // Every token in the file now belongs to sector 01 (MachineLearning)
        for t in &remapped {
            assert_eq!(t.sector.id(), DomainGroup::MachineLearning.id());
        }

        // Check syntactic role preserved in `rule` and `condition`
        let def_kw = remapped.iter().find(|t| t.text == "def").unwrap();
        assert_eq!(def_kw.rule, 'K');
        let fn_name = remapped.iter().find(|t| t.text == "forward").unwrap();
        assert_eq!(fn_name.rule, 'D');
        assert_eq!(fn_name.condition, 'F');
        assert!(fn_name.is_definition);

        let used_x = remapped.iter().rfind(|t| t.text == "x").unwrap();
        assert_eq!(used_x.rule, 'R');
        assert!(used_x.is_reference);

        let st = to_structured_domain_token(fn_name, 42);
        assert_eq!(st.to_canonical(), "01DF000042");

        // Exact source reconstruction is preserved
        let rebuilt: String = remapped.iter().map(|t| t.text.as_str()).collect();
        assert_eq!(rebuilt, ml_code);
    }
}
