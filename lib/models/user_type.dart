enum UserType {
  student, // 학생
  staff; // 교직원

  // 표시 이름으로 변환
  String get displayName {
    switch (this) {
      case UserType.student:
        return '학생';
      case UserType.staff:
        return '교직원';
    }
  }

  // 문자열을 enum으로 변환 (Firestore 데이터 읽을 때 유용)
  static UserType fromString(String value) {
    return UserType.values.firstWhere(
      (type) => type.toString().split('.').last == value,
      orElse: () => UserType.student, // 기본값은 학생
    );
  }

  // Firestore에 저장할 문자열 값
  String get value {
    return toString().split('.').last;
  }
}
