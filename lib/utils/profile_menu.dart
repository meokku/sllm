import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:swlab_sllm_app/main.dart';
import 'package:swlab_sllm_app/providers/active_chat_provider.dart';
import 'package:swlab_sllm_app/providers/chat_session_provider.dart';
import 'package:swlab_sllm_app/screens/login_screen.dart';
import 'package:swlab_sllm_app/services/auth_service.dart';

class ProfileMenuUtil {
  static final AuthService _authService = AuthService();

  // 프로필 메뉴를 표시하는 함수
  static void showProfileMenu(
    BuildContext context, {
    required GlobalKey buttonKey,
    required Function onRemove,
    required Function(OverlayEntry entry) onOverlayCreated,
    Function()? onSettingsTap,
    Function()? onLogoutTap,
  }) {
    // 사용자 정보 가져오기
    final user = _authService.currentUser;
    final userName = user?.displayName ?? '사용자 이름';
    final userEmail = user?.email;

    // 사용자 정보 로깅 (디버그용)
    print("프로필 메뉴 표시: $userName ($userEmail)");

    // GlobalKey를 사용하여 버튼의 RenderBox를 가져옵니다
    final RenderBox? renderBox =
        buttonKey.currentContext?.findRenderObject() as RenderBox?;
    if (renderBox == null) {
      print("RenderBox를 찾을 수 없습니다");
      return;
    }

    // 버튼의 위치와 크기 정보를 가져옵니다
    final Size size = renderBox.size;
    final Offset position = renderBox.localToGlobal(Offset.zero);

    // 오버레이 생성
    OverlayEntry overlayEntry = OverlayEntry(
      builder: (context) => Stack(
        children: [
          // 배경 터치 감지를 위한 투명 레이어
          Positioned.fill(
            child: GestureDetector(
              onTap: () => onRemove(),
              child: Container(color: Colors.transparent),
            ),
          ),
          // 실제 팝업 메뉴
          Positioned(
            top: position.dy + size.height + 5, // 버튼 바로 아래에 5픽셀 간격
            right:
                MediaQuery.of(context).size.width - (position.dx + size.width),
            child: Material(
              elevation: 8,
              borderRadius: BorderRadius.circular(15),
              child: Container(
                width: 220,
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(15),
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    // 헤더 (사용자 정보)
                    Padding(
                      padding: EdgeInsets.all(12),
                      child: Row(
                        children: [
                          CircleAvatar(
                            backgroundColor: Colors.grey[300],
                            child: Icon(Icons.person, color: Colors.grey[600]),
                          ),
                          SizedBox(width: 10),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  userName,
                                  style: TextStyle(fontWeight: FontWeight.bold),
                                ),
                                Text(
                                  userEmail ?? '이메일 없음',
                                  style: TextStyle(
                                    fontSize: 12,
                                    color: Colors.grey[600],
                                  ),
                                  overflow: TextOverflow.ellipsis,
                                ),
                              ],
                            ),
                          ),
                        ],
                      ),
                    ),
                    Divider(height: 1),
                    // 설정 메뉴 항목
                    InkWell(
                      onTap: () {
                        onRemove();
                        if (onSettingsTap != null) {
                          onSettingsTap();
                        } else {
                          print("설정 메뉴 클릭");
                        }
                      },
                      child: Padding(
                        padding:
                            EdgeInsets.symmetric(vertical: 12, horizontal: 16),
                        child: Row(
                          children: [
                            Icon(Icons.settings,
                                size: 20, color: Colors.grey[700]),
                            SizedBox(width: 10),
                            Text('설정'),
                          ],
                        ),
                      ),
                    ),
                    // 로그아웃 메뉴 항목
                    InkWell(
                      onTap: () async {
                        onRemove();
                        if (onLogoutTap != null) {
                          onLogoutTap();
                        } else {
                          try {
                            // 글로벌 컨텍스트 사용
                            final globalContext = navigatorKey.currentContext!;

                            // ChatSessionProvider 상태 초기화
                            Provider.of<ChatSessionProvider>(globalContext,
                                    listen: false)
                                .resetState();

                            await _authService.signOut(globalContext);
                            print("로그아웃 완료");

                            navigatorKey.currentState?.pushNamedAndRemoveUntil(
                                '/', (route) => false);
                          } catch (e) {
                            print("로그아웃 중 오류 발생: $e");
                          }
                        }
                      },
                      child: Padding(
                        padding:
                            EdgeInsets.symmetric(vertical: 12, horizontal: 16),
                        child: Row(
                          children: [
                            Icon(Icons.logout,
                                size: 20, color: Colors.grey[700]),
                            SizedBox(width: 10),
                            Text('로그아웃'),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );

    // 콜백 함수 호출하여 오버레이 전달
    onOverlayCreated(overlayEntry);

    // 오버레이 표시
    Overlay.of(context).insert(overlayEntry);
    print("오버레이 메뉴가 표시되었습니다");
  }
}
