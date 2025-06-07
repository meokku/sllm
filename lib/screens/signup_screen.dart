import 'dart:math';

import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:swlab_sllm_app/models/user_type.dart';
import 'package:swlab_sllm_app/services/auth_service.dart';
import 'package:swlab_sllm_app/theme/colors.dart';
import 'package:swlab_sllm_app/widgets/icon_text_form_field.dart';

class SignUpScreen extends StatefulWidget {
  const SignUpScreen({super.key});

  @override
  _SignUpScreenState createState() => _SignUpScreenState();
}

class _SignUpScreenState extends State<SignUpScreen> {
  final AuthService _authService = AuthService();
  final _formKey = GlobalKey<FormState>();

  String _email = '';
  String _password = '';
  String _confirmPassword = '';
  bool _isLoading = false;
  String _errorMessage = '';
  String _name = '';
  UserType _userType = UserType.student;
  String _studentId = ''; // 학번

  void _signUp() async {
    if (_formKey.currentState!.validate()) {
      setState(() {
        _isLoading = true;
        _errorMessage = '';
      });

      try {
        // 학생인 경우 학번 필수 체크
        if (_userType == UserType.student && _studentId.isEmpty) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('학생인 경우 학번을 입력해야 합니다.')),
          );
          setState(() {
            _isLoading = false;
          });
          return;
        }

        // 학생인 경우 학번 중복 검사
        if (_userType == UserType.student && _studentId.isNotEmpty) {
          bool isStudentIdExists =
              await _authService.isStudentIdAlreadyExists(_studentId);
          if (isStudentIdExists) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(content: Text('이미 등록된 학번입니다.')),
            );
            setState(() {
              _isLoading = false;
            });
            return;
          }
        }

        // 교직원인 경우 학번 필드 비우기
        if (_userType == UserType.staff) {
          _studentId = '';
        }

        await _authService.signUpWithEmailAndPassword(
          _email,
          _password,
          _name,
          _userType,
          _userType == UserType.student ? _studentId : null,
        );
        // 회원가입 성공 시 로그인 화면으로 돌아가기
        Navigator.pushNamedAndRemoveUntil(
          context,
          '/login', // 로그인 화면의 라우트 이름
          (route) => false, // 모든 이전 경로 제거
        );
      } catch (e) {
        String errorMessage = '회원가입 중 오류가 발생했습니다.';

        if (e is FirebaseAuthException) {
          if (e.code == 'email-already-in-use') {
            errorMessage = '이미 사용 중인 이메일입니다.';
          } else if (e.code == 'weak-password') {
            errorMessage = '비밀번호가 너무 약합니다.';
          }
        }

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(errorMessage)),
        );
      } finally {
        if (mounted) {
          setState(() {
            _isLoading = false;
          });
        }
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('회원가입'),
        backgroundColor: Colors.white,
        scrolledUnderElevation: 0,
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: EdgeInsets.all(16.0),
          child: Form(
            key: _formKey,
            child: Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Container(
                    width: max(
                        450,
                        MediaQuery.of(context).size.width *
                            0.4), // 두 값 중 큰 값 사용
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.grey),
                    ),
                    child: IconTextFormField(
                      icon: Icons.person_outline,
                      hintText: '이름',
                      keyboardType: TextInputType.emailAddress,
                      validator: (value) {
                        if (value == null || value.isEmpty) {
                          return '이름을 입력하세요';
                        }
                        return null;
                      },
                      onChanged: (value) => _name = value.trim(),
                    ),
                  ),
                  SizedBox(height: 10),
                  Container(
                    width: max(
                        450,
                        MediaQuery.of(context).size.width *
                            0.4), // 두 값 중 큰 값 사용
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.grey),
                    ),
                    child: IconTextFormField(
                      icon: Icons.email_outlined,
                      hintText: '이메일',
                      keyboardType: TextInputType.emailAddress,
                      validator: (value) {
                        if (value == null || value.isEmpty) {
                          return '이메일을 입력하세요';
                        }
                        if (!value.contains('@') || !value.contains('.')) {
                          return '유효한 이메일 주소를 입력하세요';
                        }
                        return null;
                      },
                      onChanged: (value) => _email = value.trim(),
                    ),
                  ),
                  SizedBox(height: 10),

                  // 비밀번호 입력 필드
                  Container(
                    width: max(
                        450,
                        MediaQuery.of(context).size.width *
                            0.4), // 두 값 중 큰 값 사용
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.grey),
                    ),
                    child: IconTextFormField(
                      icon: Icons.lock_outline,
                      hintText: '비밀번호',
                      obscureText: true,
                      hasPasswordToggle: true,
                      validator: (value) {
                        if (value == null || value.isEmpty) {
                          return '비밀번호를 입력하세요';
                        }
                        if (value.length < 6) {
                          return '비밀번호는 6자 이상이어야 합니다';
                        }
                        return null;
                      },
                      onChanged: (value) => _password = value.trim(),
                    ),
                  ),
                  SizedBox(height: 10),

                  // 비밀번호 확인 필드
                  Container(
                    width: max(
                        450,
                        MediaQuery.of(context).size.width *
                            0.4), // 두 값 중 큰 값 사용
                    decoration: BoxDecoration(
                      border: Border.all(color: Colors.grey),
                    ),
                    child: IconTextFormField(
                      icon: Icons.lock_outline,
                      hintText: '비밀번호 확인',
                      obscureText: true,
                      hasPasswordToggle: true,
                      validator: (value) {
                        if (value == null || value.isEmpty) {
                          return '비밀번호를 다시 입력하세요';
                        }
                        if (value != _password) {
                          return '비밀번호가 일치하지 않습니다';
                        }
                        return null;
                      },
                      onChanged: (value) => _confirmPassword = value.trim(),
                    ),
                  ),
                  SizedBox(height: 10),
                  if (_errorMessage.isNotEmpty)
                    Padding(
                      padding: EdgeInsets.only(bottom: 16),
                      child: Text(
                        _errorMessage,
                        style: TextStyle(color: Colors.red),
                      ),
                    ),

                  // 사용자 유형 선택 (드롭다운)
                  Container(
                    width: max(450, MediaQuery.of(context).size.width * 0.4),
                    padding: EdgeInsets.symmetric(vertical: 8),
                    child: Container(
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.grey),
                        borderRadius: BorderRadius.circular(4),
                      ),
                      padding: EdgeInsets.symmetric(horizontal: 12),
                      child: DropdownButtonHideUnderline(
                        child: DropdownButton<UserType>(
                          value: _userType,
                          isExpanded: true,
                          icon: Icon(Icons.arrow_drop_down),
                          elevation: 16,
                          onChanged: (UserType? newValue) {
                            if (newValue != null) {
                              setState(() {
                                _userType = newValue;
                              });
                            }
                          },
                          items: UserType.values
                              .map<DropdownMenuItem<UserType>>((UserType type) {
                            return DropdownMenuItem<UserType>(
                              value: type,
                              child: Text(type.displayName),
                            );
                          }).toList(),
                        ),
                      ),
                    ),
                  ),
                  SizedBox(height: 10),

                  // 학번 입력 필드 (항상 표시하되, 교직원인 경우 비활성화)
                  Container(
                    width: max(450, MediaQuery.of(context).size.width * 0.4),
                    decoration: BoxDecoration(
                      border: Border.all(
                          color: _userType == UserType.student
                              ? Colors.grey
                              : Colors.grey.withOpacity(0.5)),
                    ),
                    child: IconTextFormField(
                      icon: Icons.numbers,
                      hintText: '학번',
                      keyboardType: TextInputType.number,
                      enabled: _userType == UserType.student, // 학생인 경우에만 활성화
                      validator: (value) {
                        if (_userType == UserType.student &&
                            (value == null || value.isEmpty)) {
                          return '학번을 입력하세요';
                        }
                        return null;
                      },
                      onChanged: (value) => _studentId = value.trim(),
                    ),
                  ),
                  SizedBox(height: 32),
                  // 회원가입 버튼
                  SizedBox(
                    width: max(
                        450,
                        MediaQuery.of(context).size.width *
                            0.4), // 두 값 중 큰 값 사용
                    height: 60,
                    child: ElevatedButton(
                      onPressed: _isLoading ? null : _signUp,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: SKKUColors.green,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.zero,
                        ),
                        padding: EdgeInsets.zero,
                      ),
                      child: Padding(
                        padding: const EdgeInsets.all(8.0),
                        child: _isLoading
                            ? CircularProgressIndicator(color: Colors.white)
                            : Text(
                                '회원가입',
                                style: TextStyle(
                                  fontSize: 24, // 큰 폰트 크기 사용
                                  fontWeight: FontWeight.bold, // 볼드체로 강조
                                  color: Colors.white, // 텍스트 색상
                                ),
                              ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}
