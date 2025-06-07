import 'dart:math';

import 'package:firebase_auth/firebase_auth.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:flutter/material.dart';
import 'package:swlab_sllm_app/screens/chat_screen.dart';
import 'package:swlab_sllm_app/screens/home_screen.dart';
import 'package:swlab_sllm_app/screens/signup_screen.dart';
import 'package:swlab_sllm_app/services/auth_service.dart';
import 'package:swlab_sllm_app/theme/colors.dart';
import 'package:swlab_sllm_app/widgets/icon_text_form_field.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final AuthService _authService = AuthService();
  final _formKey = GlobalKey<FormState>();

  String _email = '';
  String _password = '';
  bool _isLoading = false;
  String _errorMessage = '';

  void _signIn() async {
    if (_formKey.currentState!.validate()) {
      setState(() {
        _isLoading = true;
        _errorMessage = '';
      });

      try {
        await _authService.signInWithEmailAndPassword(_email, _password);

        // 로그인 직후 사용자 정보 가져오기
        User? currentUser = _authService.currentUser;
        String? displayName;

        // 사용자 정보가 있으면 reload() 시도
        if (currentUser != null) {
          try {
            await currentUser.reload();
            // reload 후 최신 사용자 참조 가져오기
            currentUser = _authService.currentUser;
            displayName = currentUser?.displayName;
          } catch (e) {
            print('사용자 정보 새로고침 오류: $e');
            // 오류 발생해도 계속 진행
            displayName = currentUser?.displayName;
          }
        }

        // mounted 체크 추가
        if (mounted) {
          // 로그인 성공 후 home 화면으로 이동
          Navigator.of(context).pushReplacement(
            MaterialPageRoute(
              builder: (context) => HomeScreen(initialUserName: displayName),
            ),
          );
        }
      } on FirebaseAuthException catch (e) {
        // mounted 체크 (안전을 위해)
        if (mounted) {
          setState(() {
            _errorMessage = e.message ?? '로그인에 실패했습니다.';
          });
        }
      } finally {
        // mounted 체크
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
      body: Padding(
        padding: EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Image.asset(
                  "assets/images/SKKULLM_big.png",
                  width: max(300,
                      MediaQuery.of(context).size.width * 0.25), // 두 값 중 큰 값 사용
                ),
                SizedBox(height: 20),
                Container(
                  width: max(450,
                      MediaQuery.of(context).size.width * 0.4), // 두 값 중 큰 값 사용
                  decoration: BoxDecoration(
                    border: Border.all(
                      color: Colors.grey[300]!,
                      width: 2,
                    ),
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: Column(
                    children: [
                      // 이메일 입력 영역
                      IconTextFormField(
                        icon: Icons.email_outlined,
                        hintText: '이메일',
                        keyboardType: TextInputType.emailAddress,
                        validator: (value) {
                          if (value == null || value.isEmpty) {
                            return '이메일을 입력하세요';
                          }
                          return null;
                        },
                        onChanged: (value) => _email = value.trim(),
                      ),
                      // 구분선
                      Padding(
                        padding: const EdgeInsets.symmetric(horizontal: 10.0),
                        child: Divider(color: Colors.grey[300], thickness: 2.0),
                      ),

                      // 비밀번호 입력 영역
                      IconTextFormField(
                        icon: Icons.lock_outline,
                        hintText: '비밀번호',
                        obscureText: true,
                        hasPasswordToggle: true,
                        validator: (value) {
                          if (value == null || value.isEmpty) {
                            return '비밀번호를 입력하세요';
                          }
                          return null;
                        },
                        onChanged: (value) => _password = value.trim(),
                      ),

                      if (_errorMessage.isNotEmpty)
                        Padding(
                          padding: EdgeInsets.only(bottom: 16),
                          child: Text(
                            _errorMessage,
                            style: TextStyle(color: Colors.red),
                          ),
                        ),
                    ],
                  ),
                ),
                SizedBox(height: 24),

                // 로그인 버튼
                SizedBox(
                  width: max(450,
                      MediaQuery.of(context).size.width * 0.4), // 두 값 중 큰 값 사용
                  height: 60,
                  child: ElevatedButton(
                    onPressed: _isLoading ? null : _signIn,
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
                              'LOGIN',
                              style: TextStyle(
                                fontSize: 24, // 큰 폰트 크기 사용
                                fontWeight: FontWeight.bold, // 볼드체로 강조
                                color: Colors.white, // 텍스트 색상
                              ),
                            ),
                    ),
                  ),
                ),
                SizedBox(
                  height: 10,
                ),
                TextButton(
                  onPressed: () {
                    Navigator.pushNamed(context, '/signup');
                  },
                  style: ButtonStyle(
                    overlayColor: WidgetStateProperty.all(Colors.transparent),
                  ),
                  child: Text(
                    '계정이 없으신가요? 회원가입',
                    style: TextStyle(
                      color: SKKUColors.deepGreen,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
