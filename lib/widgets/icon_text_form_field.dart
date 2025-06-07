import 'package:flutter/material.dart';

class IconTextFormField extends StatefulWidget {
  final IconData icon;
  final String hintText;
  final bool obscureText;
  final bool hasPasswordToggle;
  final TextInputType keyboardType;
  final Function(String) onChanged;
  final String? Function(String?)? validator;
  final Color iconColor;
  final bool enabled;

  const IconTextFormField({
    super.key,
    required this.icon,
    required this.hintText,
    required this.onChanged,
    this.validator,
    this.obscureText = false,
    this.hasPasswordToggle = false,
    this.keyboardType = TextInputType.text,
    this.iconColor = Colors.grey,
    this.enabled = true,
  });

  @override
  _IconTextFormFieldState createState() => _IconTextFormFieldState();
}

class _IconTextFormFieldState extends State<IconTextFormField> {
  late bool _obscureText;

  @override
  void initState() {
    super.initState();
    _obscureText = widget.obscureText;
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(10.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          SizedBox(width: 10),
          Icon(widget.icon, color: widget.iconColor),
          SizedBox(width: 10),
          Expanded(
            child: TextFormField(
              enabled: widget.enabled,
              decoration: InputDecoration(
                border: InputBorder.none,
                hintText: widget.hintText,
                hintStyle: TextStyle(color: Colors.grey),
                // 비밀번호 토글 아이콘 추가
                suffixIcon: widget.hasPasswordToggle
                    ? IconButton(
                        icon: Icon(
                          _obscureText
                              ? Icons.visibility_off_outlined
                              : Icons.visibility_outlined,
                          color: Colors.grey,
                        ),
                        onPressed: () {
                          setState(() {
                            _obscureText = !_obscureText;
                          });
                        },
                      )
                    : null,
              ),
              obscureText: _obscureText,
              keyboardType: widget.keyboardType,
              validator: widget.validator,
              onChanged: widget.onChanged,
            ),
          ),
        ],
      ),
    );
  }
}
