import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:provider/provider.dart';
import 'package:swlab_sllm_app/screens/chat_screen.dart';
import 'package:swlab_sllm_app/screens/home_screen.dart';
import 'package:swlab_sllm_app/screens/login_screen.dart';
import 'package:swlab_sllm_app/services/auth_service.dart';

class Wrapper extends StatelessWidget {
  const Wrapper({super.key});

  @override
  Widget build(BuildContext context) {
    final authService = AuthService();

    return StreamBuilder<User?>(
      stream: authService.authStateChanges,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.active) {
          final user = snapshot.data;

          if (user != null) {
            // 사용자가 로그인 된 경우, 사용자 이름을 전달
            return HomeScreen(initialUserName: user.displayName);
          } else {
            // 로그인 되지 않은 경우
            return LoginScreen();
          }
        }

        // 연결 중이거나 에러 상태
        return Scaffold(
          body: Center(
            child: CircularProgressIndicator(),
          ),
        );
      },
    );
  }
}
