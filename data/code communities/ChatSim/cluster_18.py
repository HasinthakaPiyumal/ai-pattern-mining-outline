# Cluster 18

def build_eigen_dictionary():
    pretty_printers_dict[re.compile('^Eigen::Quaternion<.*>$')] = lambda val: EigenQuaternionPrinter(val)
    pretty_printers_dict[re.compile('^Eigen::Matrix<.*>$')] = lambda val: EigenMatrixPrinter('Matrix', val)
    pretty_printers_dict[re.compile('^Eigen::SparseMatrix<.*>$')] = lambda val: EigenSparseMatrixPrinter(val)
    pretty_printers_dict[re.compile('^Eigen::Array<.*>$')] = lambda val: EigenMatrixPrinter('Array', val)

