import 'package:flutter/material.dart';
import 'package:swlab_sllm_app/screens/chat_screen.dart';
import 'package:swlab_sllm_app/screens/login_screen.dart';
import 'package:swlab_sllm_app/theme/colors.dart';

class SelectAuthScreen extends StatelessWidget {
  const SelectAuthScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // logo
            Image.asset(
              "assets/images/SKKULLM_big.png",
              width: 500,
            ),

            SizedBox(
              height: 30,
            ),

            // login UI
            Container(
              width: 520,
              height: 240,
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(20),
                boxShadow: [
                  BoxShadow(
                    color: Colors.grey.withOpacity(0.5),
                    spreadRadius: 5,
                    blurRadius: 7,
                    offset: Offset(0, 3), // 그림자의 위치 (수평, 수직)
                  ),
                ],
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  // 로그인 버튼
                  LoginButton(
                      color: SKKUColors.blue,
                      textColor: Colors.white,
                      text: '로그인',
                      function: () {
                        Navigator.of(context).push(
                          MaterialPageRoute(
                            builder: (context) => const LoginScreen(),
                          ),
                        );
                      }),
                  SizedBox(height: 15),

                  // 회원가입 버튼
                  LoginButton(
                      color: Colors.grey[400]!,
                      textColor: Colors.black,
                      text: '회원가입',
                      function: () {}),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class LoginButton extends StatelessWidget {
  final Color color;
  final Color textColor;
  final String text;
  final Function() function;

  const LoginButton({
    required this.color,
    required this.textColor,
    required this.text,
    required this.function,
    super.key,
  });

  @override
  Widget build(BuildContext context) {
    return ElevatedButton(
      style: ElevatedButton.styleFrom(
        backgroundColor: color,
        minimumSize: const Size(450, 80),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(20),
        ),
      ),
      onPressed: function,
      child: Text(
        text,
        style: TextStyle(
          color: textColor,
          fontSize: 20,
        ),
      ),
    );
  }
}
