// 小型线性代数工具：只用于少量类别上的先验求解与病态诊断，不引入第三方依赖。

function dot(left, right) {
  let total = 0;
  for (let index = 0; index < left.length; index += 1) total += left[index] * right[index];
  return total;
}

function transposeProduct(matrix) {
  const rows = matrix.length;
  const columns = matrix[0]?.length ?? 0;
  const result = Array.from({ length: columns }, () => Array(columns).fill(0));
  for (let left = 0; left < columns; left += 1) {
    for (let right = left; right < columns; right += 1) {
      let total = 0;
      for (let row = 0; row < rows; row += 1) total += matrix[row][left] * matrix[row][right];
      result[left][right] = total;
      result[right][left] = total;
    }
  }
  return result;
}

// 高斯消元，部分选主元。奇异或接近奇异时返回 null，由调用方记录不可识别原因。
export function solveLinearSystem(matrix, vector) {
  const size = vector.length;
  if (matrix.length !== size) return null;
  const augmented = matrix.map((row, index) => [...row, vector[index]]);
  let scale = 0;
  for (const row of matrix) for (const value of row) scale = Math.max(scale, Math.abs(value));
  const threshold = Math.max(1e-14, scale * 1e-12);
  for (let column = 0; column < size; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    if (Math.abs(augmented[pivot][column]) <= threshold) return null;
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    for (let row = column + 1; row < size; row += 1) {
      const factor = augmented[row][column] / augmented[column][column];
      if (factor === 0) continue;
      for (let entry = column; entry <= size; entry += 1) {
        augmented[row][entry] -= factor * augmented[column][entry];
      }
    }
  }
  const solution = Array(size).fill(0);
  for (let row = size - 1; row >= 0; row -= 1) {
    let total = augmented[row][size];
    for (let column = row + 1; column < size; column += 1) {
      total -= augmented[row][column] * solution[column];
    }
    if (Math.abs(augmented[row][row]) <= threshold) return null;
    solution[row] = total / augmented[row][row];
  }
  return solution;
}

// 对称矩阵的循环 Jacobi 特征分解，特征值与特征向量同时返回，按特征值降序。
export function symmetricEigen(matrix) {
  const size = matrix.length;
  const working = matrix.map(row => [...row]);
  const vectors = Array.from({ length: size }, (_, row) =>
    Array.from({ length: size }, (_, column) => (row === column ? 1 : 0)));
  for (let sweep = 0; sweep < 100; sweep += 1) {
    let offDiagonal = 0;
    for (let row = 0; row < size; row += 1) {
      for (let column = row + 1; column < size; column += 1) {
        offDiagonal += working[row][column] ** 2;
      }
    }
    if (offDiagonal <= 1e-30) break;
    for (let left = 0; left < size; left += 1) {
      for (let right = left + 1; right < size; right += 1) {
        const value = working[left][right];
        if (Math.abs(value) <= 1e-18) continue;
        const theta = (working[right][right] - working[left][left]) / (2 * value);
        const sign = theta >= 0 ? 1 : -1;
        const tangent = sign / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const cosine = 1 / Math.sqrt(tangent * tangent + 1);
        const sine = tangent * cosine;
        for (let row = 0; row < size; row += 1) {
          const leftValue = working[row][left];
          const rightValue = working[row][right];
          working[row][left] = cosine * leftValue - sine * rightValue;
          working[row][right] = sine * leftValue + cosine * rightValue;
          const vectorLeft = vectors[row][left];
          const vectorRight = vectors[row][right];
          vectors[row][left] = cosine * vectorLeft - sine * vectorRight;
          vectors[row][right] = sine * vectorLeft + cosine * vectorRight;
        }
        for (let column = 0; column < size; column += 1) {
          const leftValue = working[left][column];
          const rightValue = working[right][column];
          working[left][column] = cosine * leftValue - sine * rightValue;
          working[right][column] = sine * leftValue + cosine * rightValue;
        }
      }
    }
  }
  const order = working.map((row, index) => index)
    .sort((left, right) => working[right][right] - working[left][left]);
  return {
    values: order.map(index => working[index][index]),
    vectors: order.map(index => vectors.map(row => row[index]))
  };
}

// 条件数为最大与最小奇异值之比；秩亏损时条件数为 null，不伪造有限值。
export function matrixCondition(matrix) {
  if (!matrix.length || !matrix[0]?.length) {
    return { value: null, rank: 0, singularValues: [], rankDeficient: true };
  }
  const eigenvalues = symmetricEigen(transposeProduct(matrix))
    .values.map(value => Math.max(0, value));
  const largest = eigenvalues[0] ?? 0;
  const smallest = eigenvalues[eigenvalues.length - 1] ?? 0;
  const rank = largest > 0 ? eigenvalues.filter(value => value > largest * 1e-12).length : 0;
  const singularValues = eigenvalues.map(value => Math.sqrt(value));
  const value = largest > 0 && smallest > largest * 1e-24
    ? Math.sqrt(largest / smallest)
    : null;
  return { value, rank, singularValues, rankDeficient: rank < matrix[0].length };
}
