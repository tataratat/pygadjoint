#pragma once
namespace gismo::expr {

/*
Expression for the Frobenius matrix product between a matrix of matrices and a
single matrix with cardinality 1
*/
template <typename E1, typename E2>
class frobenius_prod_expr : public _expr<frobenius_prod_expr<E1, E2>> {
public:
  typedef typename E1::Scalar Scalar;
  enum {
    ScalarValued = 0,
    Space = E1::Space,
    ColBlocks = 0 // E1::ColBlocks || E2::ColBlocks
  };

private:
  typename E1::Nested_t _u;
  typename E2::Nested_t _v;

  mutable gsMatrix<Scalar> res;

public:
  frobenius_prod_expr(_expr<E1> const &u, _expr<E2> const &v) : _u(u), _v(v) {
    // todo: add check() functions, which will evaluate expressions on an empty
    // matrix (no points) to setup initial dimensions ???
    GISMO_ASSERT(_u.rows() == _v.rows(), "Wrong dimensions "
                                             << _u.rows() << "!=" << _v.rows()
                                             << " in % operation");
    GISMO_ASSERT(_u.cols() == _v.cols(), "Wrong dimensions "
                                             << _u.cols() << "!=" << _v.cols()
                                             << " in %operation");
  }

  const gsMatrix<Scalar> &eval(const index_t k) const {
    // Evaluate Expressions and cardinality
    const index_t u_r = _u.rows();
    const index_t u_c = _u.cols();

    // Cardinality impl refers to the cols im a matrix
    auto A = _u.eval(k);
    auto B = _v.eval(k);
    const index_t A_rows = A.rows() / u_r;
    const index_t A_cols = A.cols() / u_c;
    GISMO_ASSERT(
        _v.cardinality() == 1,
        "Expression is only for second expressions with cardinality 1");
    GISMO_ASSERT(
        (_u.cols() == _v.cols()) && (_u.rows() == _v.rows()),
        "Both expressions need to be same size, but : "
            << "\n_u.cols() : " << _u.cols() << "\n_u.rows() : " << _u.rows()
            << "\n_v.cols() : " << _v.cols() << "\n_v.rows() : " << _v.rows()
            << ((_u.cols() == _v.cols()) ? "Rows" : "Cols") << " are different"
            << ((_u.rows() == _v.rows()) ? "Or are they? "
                                         : "Yep different! "));
    GISMO_ASSERT(B.size() == _v.rows() * _v.cols(),
                 "RHS expression contains more than one matrix");
    res.resize(A_rows, A_cols);
    for (index_t i = 0; i < A_rows; ++i)
      for (index_t j = 0; j < A_cols; ++j)
        res(i, j) =
            (A.block(i * u_r, j * u_c, u_r, u_c).array() * B.array()).sum();
    return res;
  }

  index_t rows() const { return 1; }
  index_t cols() const { return 1; }

  void parse(gsExprHelper<Scalar> &evList) const {
    _u.parse(evList);
    _v.parse(evList);
  }

  const gsFeSpace<Scalar> &rowVar() const { return _u.rowVar(); }
  const gsFeSpace<Scalar> &colVar() const { return _u.colVar(); } // overwrite

  void print(std::ostream &os) const {
    os << "(";
    _u.print(os);
    os << " % ";
    _v.print(os);
    os << ")";
  }
};

template <typename E1, typename E2>
EIGEN_STRONG_INLINE frobenius_prod_expr<E1, E2> const
frobenius(_expr<E1> const &u, _expr<E2> const &v) {
  return frobenius_prod_expr<E1, E2>(u, v);
}

/*
 * Custom expression for multiplication of two scalar valued functions, where
 * one is a vector and one is a transposed vector
 */
template <typename E1, typename E2>
class multiplication_expr : public _expr<multiplication_expr<E1, E2>> {
public:
  typedef typename E1::Scalar Scalar;
  enum {
    ScalarValued = 0,
    Space = E1::Space + E2::Space, // Should always be 3
    ColBlocks = 0                  // E1::ColBlocks || E2::ColBlocks
  };

private:
  typename E1::Nested_t _u;
  typename E2::Nested_t _v;

  mutable gsMatrix<Scalar> res;

public:
  multiplication_expr(_expr<E1> const &u, _expr<E2> const &v) : _u(u), _v(v) {
    // todo: add check() functions, which will evaluate expressions on an empty
    // matrix (no points) to setup initial dimensions ???
    GISMO_ASSERT(_u.rows() == 1 && _u.cols() == 1,
                 "Wrong dimensions for first argument, rows : "
                     << _u.rows() << " cols :" << _u.cols()
                     << " in * operation");
    GISMO_ASSERT(_u.rows() == 1 && _u.cols() == 1,
                 "Wrong dimensions for first argument, rows : "
                     << _v.rows() << " cols :" << _v.cols()
                     << " in * operation");
    GISMO_ASSERT((_u.isVector() || _v.isVector()) &&
                     (_u.isVectorTr() || _v.isVectorTr()),
                 "Wrong dimensions for first argument, rows : "
                     << _v.rows() << " cols :" << _v.cols()
                     << " in * operation");
    GISMO_ASSERT((E1::ColBlocks == 0) && (E2::ColBlocks == 0),
                 "Expected scalar valued per entry Expression");
    GISMO_ASSERT((E1::Space + E2::Space == 3),
                 "Expected scalar valued per entry Expression");
  }

  const gsMatrix<Scalar> &eval(const index_t k) const {
    // Cardinality impl refers to the cols im a matrix
    auto row_values = _u.isVector() ? _u.eval(k) : _v.eval(k);
    auto col_values = _u.isVectorTr() ? _u.eval(k) : _v.eval(k);

    const index_t n_rows = row_values.rows();
    const index_t n_cols = col_values.cols();

    res.resize(n_rows, n_cols);
    for (index_t i = 0; i < n_rows; ++i)
      for (index_t j = 0; j < n_cols; ++j)
        res(i, j) = row_values(i) * col_values(j);
    return res;
  }

  index_t rows() const { return 1; }
  index_t cols() const { return 1; }

  void parse(gsExprHelper<Scalar> &evList) const {
    _u.parse(evList);
    _v.parse(evList);
  }

  const gsFeSpace<Scalar> &rowVar() const { return _u.rowVar(); }
  const gsFeSpace<Scalar> &colVar() const { return _v.colVar(); } // overwrite

  void print(std::ostream &os) const {
    os << "(";
    _u.print(os);
    os << " (x) ";
    _v.print(os);
    os << ")";
  }
};

template <typename E1, typename E2>
EIGEN_STRONG_INLINE multiplication_expr<E1, E2> const
multiply(_expr<E1> const &u, _expr<E2> const &v) {
  return multiplication_expr<E1, E2>(u, v);
}

} // namespace gismo::expr
