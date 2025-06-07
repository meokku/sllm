import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:swlab_sllm_app/main.dart';
import 'package:swlab_sllm_app/models/user_type.dart';
import 'package:swlab_sllm_app/providers/active_chat_provider.dart';
import 'package:swlab_sllm_app/providers/chat_session_provider.dart';
import 'package:swlab_sllm_app/screens/login_screen.dart';

class AuthService {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;

  // 현재 사용자 상태 스트림
  Stream<User?> get authStateChanges => _auth.authStateChanges();

  // 이메일/비밀번호 회원가입
  Future<UserCredential> signUpWithEmailAndPassword(
    String email,
    String password,
    String name,
    UserType userType,
    String? studentId,
  ) async {
    // 사용자 계정 생성
    UserCredential userCredential = await _auth.createUserWithEmailAndPassword(
      email: email,
      password: password,
    );

    // 기본 프로필 업데이트
    await userCredential.user!.updateDisplayName(name);

    // Firestore에 저장할 사용자 정보 맵 생성
    Map<String, dynamic> userData = {
      'name': name,
      'email': email,
      'userType': userType.value,
      'createdAt': FieldValue.serverTimestamp(),
    };

    // 학생인 경우 학번 정보 추가
    if (userType == UserType.student &&
        studentId != null &&
        studentId.isNotEmpty) {
      userData['studentId'] = studentId;
    }

    // Firestore에 사용자 정보 저장
    await _firestore
        .collection('users')
        .doc(userCredential.user!.uid)
        .set(userData);

    // 사용자 정보 리로드
    await userCredential.user?.reload();

    return userCredential;
  }

  // 이메일/비밀번호 로그인
  Future<UserCredential?> signInWithEmailAndPassword(
      String email, String password) async {
    try {
      final userCredential = await _auth.signInWithEmailAndPassword(
        email: email,
        password: password,
      );

      // 로그인 후 즉시 상태 초기화
      if (userCredential.user != null) {
        // 글로벌 컨텍스트 사용 (필요한 경우)
        final context = navigatorKey.currentContext;
        if (context != null) {
          Provider.of<ChatSessionProvider>(context, listen: false).resetState();
          Provider.of<ActiveChatProvider>(context, listen: false)
              .clearActiveChat();
        }
      }

      return userCredential;
    } on FirebaseAuthException catch (e) {
      print('로그인 오류: ${e.message}');
      rethrow;
    }
  }

  // 로그아웃
  Future<void> signOut(BuildContext context) async {
    try {
      // 현재 사용자 확인
      final User? currentUser = _auth.currentUser;
      if (currentUser == null) {
        print('이미 로그아웃된 상태입니다.');
        return;
      }

      // Firebase 로그아웃
      await _auth.signOut();

      // 화면 전환 후 상태 초기화
      WidgetsBinding.instance.addPostFrameCallback((_) {
        try {
          final chatSessionProvider =
              Provider.of<ChatSessionProvider>(context, listen: false);
          final activeChatProvider =
              Provider.of<ActiveChatProvider>(context, listen: false);

          chatSessionProvider.resetState();
          activeChatProvider.clearActiveChat();
        } catch (e) {
          print('Provider 상태 초기화 오류: $e');
        }
      });

      print('로그아웃 성공');
    } catch (e) {
      print('로그아웃 오류: $e');
      rethrow;
    }
  }

  // reload 함수
  Future<void> reloadUser() async {
    final user = _auth.currentUser;
    if (user != null) {
      await user.reload();
    }
  }

  /// 학번 중복 검사
  Future<bool> isStudentIdAlreadyExists(String studentId) async {
    try {
      // 학번으로 사용자 조회 쿼리
      final querySnapshot = await _firestore
          .collection('users')
          .where('studentId', isEqualTo: studentId)
          .get();

      // 결과가 비어있지 않으면 이미 존재함
      return querySnapshot.docs.isNotEmpty;
    } catch (e) {
      print('학번 중복 검사 중 오류 발생: $e');
      rethrow;
    }
  }

  // 사용자 상태 변경 시 Provider 업데이트를 위한 메서드
  void setupAuthStateListener(BuildContext context) {
    _auth.authStateChanges().listen((User? user) {
      if (user != null) {
        // 로그인 상태
        Provider.of<ChatSessionProvider>(context, listen: false).resetState();
      } else {
        // 로그아웃 상태
        Provider.of<ChatSessionProvider>(context, listen: false).resetState();
        Provider.of<ActiveChatProvider>(context, listen: false)
            .clearActiveChat();
      }
    });
  }

  // 현재 사용자 가져오기
  User? get currentUser => _auth.currentUser;
}
